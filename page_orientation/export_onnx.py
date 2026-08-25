"""Export a fine-tuned page-orientation checkpoint (best.pt) to ONNX.

Kept separate from training so that an export failure can never lose a
finished training run. Verifies torch/ONNX output parity on random inputs
when onnxruntime is available.

Usage: python export_onnx.py [path/to/config.yaml] [path/to/best.pt]
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from doctr.models.classification import mobilenet_v3_small_page_orientation

OPSET = 18


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    out_dir = (cfg_path.parent / str(cfg.out_dir)).resolve()
    ckpt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir / "best.pt"
    onnx_path = out_dir / "page_orientation.onnx"
    size = int(cfg.model.input_size)

    # pretrained=False: weights come from the checkpoint, no download needed
    model = mobilenet_v3_small_page_orientation(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )
    print(f"exported {ckpt_path} -> {onnx_path}")

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, parity check skipped")
        return

    sess = ort.InferenceSession(str(onnx_path))
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"parity check: max|torch - onnx| = {diff:.2e}", "(OK)" if diff < 1e-4 else "(MISMATCH!)")


if __name__ == "__main__":
    main()
