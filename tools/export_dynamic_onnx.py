import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import onnx

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


CHECKPOINT_PATH = "models/checkpoint_iter_370000.pth"

OUTPUT_ONNX = "pose_model_dynamic.onnx"

BASE_HEIGHT = 544
BASE_WIDTH = 968

NUM_REFINEMENT_STAGES = 1


def main():
    device = torch.device("cpu")

    # =========================
    # Load model
    # =========================
    net = PoseEstimationWithMobileNet(
        num_refinement_stages=NUM_REFINEMENT_STAGES
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    load_state(net, checkpoint)

    net.eval()

    # =========================
    # Dummy input
    # =========================
    dummy_input = torch.randn(
        1,
        3,
        BASE_HEIGHT,
        BASE_WIDTH,
        dtype=torch.float32
    )

    # =========================
    # Export dynamic ONNX
    # =========================
    torch.onnx.export(
        net,
        dummy_input,
        OUTPUT_ONNX,

        export_params=True,
        opset_version=17,
        do_constant_folding=True,

        input_names=["input"],
        output_names=["heatmaps", "pafs"],

        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "heatmaps": {
                0: "batch_size"
            },
            "pafs": {
                0: "batch_size"
            }
        }
    )

    print(f"Dynamic ONNX exported to: {OUTPUT_ONNX}")

    # =========================
    # Verify ONNX
    # =========================
    model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(model)

    print("ONNX model verified successfully.")

    # Print input shapes
    for inp in model.graph.input:
        print(inp)


if __name__ == "__main__":
    main()