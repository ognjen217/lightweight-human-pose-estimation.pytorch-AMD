#!/usr/bin/env python3
"""Export graph-clean ONNX variants for MIGraphX experiments.

This exporter is intentionally separate from export_dynamic_onnx.py. It can
optionally skip the model's final DeQuantStub during export and optionally run
onnxsim. The default output still exposes separate heatmap and PAF tensors, so
existing validation/postprocessing code can keep the same assumptions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import onnx

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoint_iter_370000.pth")
    parser.add_argument("--output", default="models/onnx/pose_model_clean_bdyn.onnx")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=968)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-refinement-stages", type=int, default=1)
    parser.add_argument(
        "--without-dequant",
        action="store_true",
        help="Export final heatmap/PAF tensors before the model's DeQuantStub.",
    )
    parser.add_argument(
        "--concat-outputs",
        action="store_true",
        help=(
            "Export a single 57-channel output by concatenating heatmaps and PAFs "
            "after the model heads. This is for profiling only; it changes the "
            "output contract expected by existing postprocessing code."
        ),
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxsim after export when the package is installed.",
    )
    parser.add_argument(
        "--simplified-output",
        default=None,
        help="Optional path for the simplified ONNX model. Defaults to *_sim.onnx.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


class ExportWrapper(torch.nn.Module):
    def __init__(self, net: PoseEstimationWithMobileNet, concat_outputs: bool = False):
        super().__init__()
        self.net = net
        self.concat_outputs = concat_outputs

    def forward(self, x):
        heatmaps, pafs = self.net(x)
        if self.concat_outputs:
            return torch.cat([heatmaps, pafs], dim=1)
        return heatmaps, pafs


def export_model(args: argparse.Namespace) -> Path:
    device = torch.device("cpu")

    net = PoseEstimationWithMobileNet(num_refinement_stages=args.num_refinement_stages)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_state(net, checkpoint)
    net.eval()
    net.export_without_dequant = args.without_dequant

    wrapped = ExportWrapper(net, concat_outputs=args.concat_outputs).eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        dtype=torch.float32,
        device=device,
    )

    if args.concat_outputs:
        output_names = ["heatmaps_pafs"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "heatmaps_pafs": {0: "batch_size"},
        }
    else:
        output_names = ["heatmaps", "pafs"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
            "pafs": {0: "batch_size"},
        }

    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print(f"Exported ONNX: {output_path}")

    return output_path


def maybe_simplify(args: argparse.Namespace, onnx_path: Path) -> Path | None:
    if not args.simplify:
        return None

    try:
        import onnxsim
    except ImportError as exc:
        raise SystemExit(
            "onnxsim is not installed. Install it or rerun without --simplify."
        ) from exc

    simplified_output = (
        Path(args.simplified_output)
        if args.simplified_output
        else onnx_path.with_name(f"{onnx_path.stem}_sim{onnx_path.suffix}")
    )
    simplified_output.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    simplified_model, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim returned check=False; simplified model is not valid.")

    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model, str(simplified_output))
    print(f"Simplified ONNX: {simplified_output}")

    return simplified_output


def write_report(args: argparse.Namespace, exported: Path, simplified: Path | None) -> None:
    if not args.report_json:
        return

    report = {
        "checkpoint": args.checkpoint,
        "exported": str(exported),
        "simplified": str(simplified) if simplified else None,
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "num_refinement_stages": args.num_refinement_stages,
        "without_dequant": args.without_dequant,
        "concat_outputs": args.concat_outputs,
        "opset": args.opset,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote export report: {report_path}")


def main() -> None:
    args = parse_args()
    exported = export_model(args)
    simplified = maybe_simplify(args, exported)
    write_report(args, exported, simplified)


if __name__ == "__main__":
    main()
