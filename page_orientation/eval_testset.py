"""Evaluate a page-orientation checkpoint on a small hand-picked test set.

Every image in TESTSET_DIR is assumed to be stored upright; it is evaluated
in all four orientations (4N predictions). Prints a per-image table and the
accumulated per-angle accuracy. A row failing in a consistent, shifted way
usually means the source image itself is not upright.

Usage: python eval_testset.py [testset_dir] [checkpoint]
python eval_testset.py /mnt/DATA2/dorin/images/htr /mnt/DATA2/dorin/handwriting-recognition/page_orientation/checkpoints_1/best_1.pt
python eval_testset.py /mnt/DATA2/dorin/handwriting-recognition/data/kaliningrad_essays /mnt/DATA2/dorin/handwriting-recognition/page_orientation/checkpoints_1/best_1.pt
"""
import sys
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageFile
from torchvision import transforms as T
from doctr.models.classification import mobilenet_v3_small_page_orientation

ImageFile.LOAD_TRUNCATED_IMAGES = True

TESTSET_DIR = sys.argv[1] if len(sys.argv) > 1 else "testset"
CKPT = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "best.pt")
METRICS_JSON = Path(__file__).parent / "eval_metrics.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [0, -90, 180, 90]
K_TO_CLASS = [0, 3, 2, 1]
PIL_ROT = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]
ANGLE_NAMES = ["0", "+90", "180", "-90"]  # by k

preprocess = T.Compose([
    T.Resize(511, max_size=512),
    T.CenterCrop(512),
    T.ToTensor(),
    T.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
])


@torch.no_grad()
def predict(model, img):
    logits = model(preprocess(img).unsqueeze(0).to(DEVICE))[0]
    probs = logits.softmax(-1)
    idx = int(probs.argmax())
    return idx, float(probs[idx])


def main():
    model = mobilenet_v3_small_page_orientation(pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.to(DEVICE).eval()

    paths = sorted(p for p in Path(TESTSET_DIR).rglob("*")
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert paths, f"no images found in {TESTSET_DIR}"
    print(f"checkpoint: {CKPT}\n{len(paths)} images x 4 rotations\n")

    per_k_correct = [0] * 4
    total_correct = 0
    per_image = {}
    name_w = max(len(p.name) for p in paths)

    header = f"{'image':<{name_w}} | " + " | ".join(f"rot {a:>4}" for a in ANGLE_NAMES)
    print(header)
    print("-" * len(header))

    for p in paths:
        src = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        cells, row = [], {}
        for k in range(4):
            img = src.transpose(PIL_ROT[k]) if PIL_ROT[k] is not None else src
            idx, conf = predict(model, img)
            ok = idx == K_TO_CLASS[k]
            per_k_correct[k] += ok
            total_correct += ok
            mark = "ok " if ok else "MISS"
            cells.append(f"{mark} {CLASSES[idx]:>4} {conf:.2f}")
            row[f"rot_{ANGLE_NAMES[k]}"] = {"ok": bool(ok), "pred": CLASSES[idx], "conf": round(conf, 4)}
        per_image[str(p.relative_to(TESTSET_DIR))] = row
        print(f"{p.name:<{name_w}} | " + " | ".join(cells))

    n = len(paths)
    print("-" * len(header))
    print(f"accuracy: total {total_correct}/{4 * n} = {total_correct / (4 * n):.3f}   "
          + "   ".join(f"rot {a}: {c}/{n}" for a, c in zip(ANGLE_NAMES, per_k_correct)))

    # append the run to the metrics store
    store = {}
    if METRICS_JSON.exists():
        with open(METRICS_JSON, encoding="utf-8") as f:
            store = json.load(f)
    run_key = f"{Path(TESTSET_DIR).name} | ckpt={Path(CKPT).name}"
    store[run_key] = {
        "images": n,
        "accuracy": round(total_correct / (4 * n), 4),
        "per_angle": {a: f"{c}/{n}" for a, c in zip(ANGLE_NAMES, per_k_correct)},
        "per_image": per_image,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    print(f"metrics saved to {METRICS_JSON} under key '{run_key}'")


if __name__ == "__main__":
    main()
