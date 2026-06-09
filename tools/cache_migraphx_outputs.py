#!/usr/bin/env python3
"""
Cache MIGraphX heatmap/PAF outputs for COCO images without importing torch.
This avoids MIGraphX <-> PyTorch ROCm initialization conflicts.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import migraphx
import numpy as np
from pycocotools.coco import COCO

BASE_HEIGHT = 544
BASE_WIDTH = 968
STRIDE = 8


def normalize(img, img_mean=(128, 128, 128), img_scale=1/256):
    img = img.astype(np.float32)
    img = (img - np.array(img_mean, dtype=np.float32)) * img_scale
    return img


def pad_width(img, stride, pad_value=(0, 0, 0), min_dims=(BASE_HEIGHT, BASE_WIDTH)):
    h, w, _ = img.shape
    min_h, min_w = min_dims
    pad = [0, 0, 0, 0]  # top, left, bottom, right
    pad[2] = max(min_h - h, 0)
    pad[3] = max(min_w - w, 0)

    padded_img = cv2.copyMakeBorder(
        img,
        pad[0], pad[2], pad[1], pad[3],
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded_img, pad


def prepare_input(img):
    normed_img = normalize(img)
    height, width, _ = normed_img.shape
    ratio = min(BASE_HEIGHT / height, BASE_WIDTH / width)
    scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    padded_img, pad = pad_width(scaled_img, STRIDE, (0, 0, 0), (BASE_HEIGHT, BASE_WIDTH))
    n_input = padded_img.transpose(2, 0, 1)[None, ...]
    n_input = np.ascontiguousarray(n_input)
    return n_input, pad, width, height


def run_migraphx(model, n_input):
    param_type = str(model.get_parameter_shapes()['input'].type())
    if 'half' in param_type:
        n_input = n_input.astype(np.float16)
    elif 'bfloat' in param_type or 'float' in param_type:
        n_input = n_input.astype(np.float32)
    return model.run({'input': n_input})


def decode_outputs(raw_results, pad):
    # same convention as previous benchmark: last two outputs are heatmaps and PAFs
    heatmaps = np.transpose(np.asarray(raw_results[-2]).squeeze().astype(np.float32), (1, 2, 0))
    pafs = np.transpose(np.asarray(raw_results[-1]).squeeze().astype(np.float32), (1, 2, 0))

    scaled_pad = [p // STRIDE for p in pad]
    heatmaps = heatmaps[
        scaled_pad[0]:heatmaps.shape[0] - scaled_pad[2],
        scaled_pad[1]:heatmaps.shape[1] - scaled_pad[3],
        :,
    ]
    pafs = pafs[
        scaled_pad[0]:pafs.shape[0] - scaled_pad[2],
        scaled_pad[1]:pafs.shape[1] - scaled_pad[3],
        :,
    ]
    return heatmaps, pafs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--images-folder', required=True)
    ap.add_argument('--output-dir', default='outputs/cached_migraphx_val')
    ap.add_argument('--max-images', type=int, default=5000)
    ap.add_argument('--start-index', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading MIGraphX model first: {args.model}')
    model = migraphx.load(args.model)
    print('MIGraphX load OK')

    coco = COCO(args.labels)
    image_ids = sorted(coco.getImgIds())
    image_ids = image_ids[args.start_index:]
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]

    manifest = []
    for idx, image_id in enumerate(image_ids, 1):
        info = coco.loadImgs([image_id])[0]
        img_path = os.path.join(args.images_folder, info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'WARNING: cannot read {img_path}; skipping')
            continue

        n_input, pad, orig_w, orig_h = prepare_input(img)
        raw = run_migraphx(model, n_input)
        heatmaps, pafs = decode_outputs(raw, pad)

        cache_name = f'{image_id:012d}.npz'
        cache_path = out_dir / cache_name
        np.savez_compressed(
            cache_path,
            heatmaps=heatmaps.astype(np.float16),
            pafs=pafs.astype(np.float16),
            image_id=np.int64(image_id),
            orig_w=np.int32(orig_w),
            orig_h=np.int32(orig_h),
            file_name=info['file_name'],
        )
        manifest.append({'image_id': int(image_id), 'file_name': info['file_name'], 'cache': cache_name})
        if idx % 20 == 0 or idx == 1:
            print(f'Cached {idx}/{len(image_ids)}: {info["file_name"]}') 

    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'labels': args.labels, 'images_folder': args.images_folder, 'items': manifest}, f, indent=2)

    print(f'Done. Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
