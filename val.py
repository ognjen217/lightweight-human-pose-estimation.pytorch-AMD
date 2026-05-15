import argparse
import cv2
import json
import math
import numpy as np
np.float = float
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
torch.set_num_threads(16)
from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state

base_height = 544
base_width = 968

def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(net, img, scales, base_height, base_width, stride, quantization_type='fp32', pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [min(base_height / height, base_width / width)]
    
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    if quantization_type in ['int8', 'mixed_fp32', 'mixed_fp16']:
        device = torch.device('cpu')
        input_dtype = torch.float32 
        if quantization_type == 'mixed_fp16':
            input_dtype = torch.float16
    else:
        device = torch.device('cuda')
        input_dtype = torch.float16 if quantization_type == 'fp16' else \
                      torch.bfloat16 if quantization_type == 'bf16' else torch.float32
    
    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        min_dims = [base_height, base_width]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).to(device).to(input_dtype)
        
        with torch.no_grad():
            stages_output = net(tensor_img)
        
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().float().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().float().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs

def evaluate(labels, output_name, images_folder, net, multiscale=False, visualize=False, quantization_type='fp32'):
    net = net.eval()
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    for i, sample in enumerate(dataset):
        if i < 134:
            continue
        file_name = sample['file_name']
        img = sample['img']
        
        if i % 10 == 0:
            print(f"Processing image {i}/5000: {file_name}")
        
        avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, base_width, stride, quantization_type)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return
    print(f"\n--- Inference Complete. Writing results to {output_name}... ---")
    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)
    print("--- Starting COCO Evaluation ---")
    run_coco_eval(labels, output_name)
    print("--- Evaluation Finished! ---")

def load_model(args):
    net = PoseEstimationWithMobileNet(num_refinement_stages=args.num_refinement_stages)
    net.eval()
    if args.quantization in ['int8', 'mixed_fp32', 'mixed_fp16']:
        checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location=torch.device('cpu'))
        load_state(net, checkpoint)
        net.eval()
        torch.backends.quantized.engine = 'fbgemm'
        net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        net.fuse_model()
        if args.quantization in ['mixed_fp32', 'mixed_fp16']:
            layers_to_skip = [net.model[0], net.model[1], net.model[2]]
            net.is_mixed = True
            for layer in layers_to_skip:
                layer.qconfig = None 

        torch.quantization.prepare(net, inplace=True)
        calibrate_model(net, args)
        torch.quantization.convert(net, inplace=True)

        if args.quantization == 'mixed_fp16':
            print("--- Manually casting first 3 layers to FP16 ---")
            for layer in layers_to_skip:
                layer.half()
        print("Loaded calibrated INT8 weights successfully.")
        return net
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cuda'))
        load_state(net, checkpoint)
        net = net.to(torch.device('cuda'))
        if args.quantization == 'fp16':
            net.half()
            print("Running in FP16 (Half Precision).")
        elif args.quantization == 'bf16':
            net.to(torch.bfloat16)
            print("Running in BF16 (BFloat16).")
        
        return net.eval()

def calibrate_model(model, args):
    print("--- Starting Calibration (Static PTQ) ---")
    dataset = CocoValDataset(args.labels, args.images_folder)
    with torch.no_grad():
        for i in range(min(135, len(dataset))):
            img = dataset[i]['img']
            img_resized = cv2.resize(img, (base_width, base_height))
            img_mean = 128
            img_scale = 1/256
            normalized_img = (img.astype(np.float32) - img_mean) * img_scale
            input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).float()
            model(input_tensor)
            if i % 10 == 0:
                print(f"Calibrating image {i}/135...")
    print("--- Calibration Complete ---")

def export_to_onnx(net, output_path="pose_model.onnx"):
    net.eval().cpu()
    # Get the model's dtype from its parameters
    model_dtype = next(net.parameters()).dtype
    dummy_input = torch.randn(1, 3, base_height, base_width, dtype=model_dtype)
    torch.onnx.export(net, dummy_input, output_path, 
                      input_names=['input'], 
                      output_names=['stage_heatmaps', 'stage_pafs'],
                      opset_version=11)
    print(f"Model exported to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    parser.add_argument('--num-refinement-stages', type=int, help='preformance')
    parser.add_argument('--quantization', type=str, default='fp32', choices=['fp32', 'fp16', 'int8', 'bf16', 'mixed_fp32', 'mixed_fp16'])
    parser.add_argument('--export', action='store_true', help='export the model before evaluation')
    parser.add_argument('--export-name', type=str, default='pose_model.onnx', help='output filename for exported model')

    args = parser.parse_args()

    net = load_model(args)
    

    if args.export:
        export_to_onnx(net, output_path=args.export_name)
        # Move model back to CUDA if it was FP16
        if args.quantization == 'fp16':
            net = net.to(torch.device('cuda'))

    evaluate(args.labels, args.output_name, args.images_folder, net, 
             args.multiscale, args.visualize, args.quantization)


