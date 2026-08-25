"""Fine-tuning of the docTR page-orientation classifier (4 classes: 0/90/180/270).

Training data: a directory of page images in the upright orientation (searched
recursively). Rotation labels are generated on the fly: each sample is rotated
by a random multiple of 90 degrees, the applied rotation defines the target class.

Validation: every image is evaluated in all four orientations. The confusion
matrix is accumulated over the full epoch; accuracy and macro precision/recall/F1
are computed from the accumulated counts.

Usage: python train.py [path/to/config.yaml]
"""
import sys
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile, UnidentifiedImageError

# HWR200 contains a few truncated JPEGs; decode what is available instead of raising
ImageFile.LOAD_TRUNCATED_IMAGES = True

_bad_files = set()


def open_rgb(path):
    """Open an image or return None for unreadable files (broken header, empty file)."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        if path not in _bad_files:
            _bad_files.add(path)
            print(f"warning: skipping unreadable image {path}")
        return None
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from doctr.models.classification import mobilenet_v3_small_page_orientation

# docTR class order: [0, -90, 180, 90] (page rotation angle, CCW-positive).
CLASSES = [0, -90, 180, 90]
NUM_CLASSES = len(CLASSES)
# PIL Transpose.ROTATE_{90k} rotates 90k degrees CCW; K_TO_CLASS[k] is the
# index of the resulting angle in CLASSES: 0->0, +90->3, 180->2, -90(=270 CCW)->1.
K_TO_CLASS = [0, 3, 2, 1]
PIL_ROT = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def build_transforms(cfg):
    size = int(cfg.model.input_size)
    preprocess = T.Compose([
        T.Resize(size - 1, max_size=size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=list(cfg.model.mean), std=list(cfg.model.std)),
    ])
    color_aug = T.ColorJitter(brightness=cfg.aug.brightness, contrast=cfg.aug.contrast,
                              saturation=cfg.aug.saturation, hue=cfg.aug.hue)
    return preprocess, color_aug


# --- data ----------------------------------------------------------------
class TrainPages(Dataset):
    """Random orientation per access: over an epoch each page is seen at varying angles."""

    def __init__(self, paths, preprocess, color_aug):
        self.paths = paths
        self.preprocess = preprocess
        self.color_aug = color_aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = open_rgb(self.paths[i])
        if img is None:
            return None
        img = self.color_aug(img)
        k = random.randrange(4)
        if PIL_ROT[k] is not None:
            img = img.transpose(PIL_ROT[k])
        return self.preprocess(img), K_TO_CLASS[k]


class ValPages(Dataset):
    """Deterministic evaluation set: each page in all four orientations."""

    def __init__(self, paths, preprocess):
        self.items = [(p, k) for p in paths for k in range(4)]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, k = self.items[i]
        img = open_rgb(path)
        if img is None:
            return None
        if PIL_ROT[k] is not None:
            img = img.transpose(PIL_ROT[k])
        return self.preprocess(img), K_TO_CLASS[k]


def build_scheduler(opt, total_steps, warmup_frac, min_lr_frac):
    """Per-step LR schedule: linear warmup, then cosine decay to lr * min_lr_frac."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def collate_skip_broken(batch):
    """Drop unreadable samples (None) from the batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


# --- metrics ----------------------------------------------------------------
@torch.no_grad()
def confusion(model, loader, device):
    """Confusion matrix accumulated over the full dataset. cm[i, j] = #{true i, pred j}."""
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for batch in tqdm(loader, desc="val", leave=False):
        if batch is None:
            continue
        x, y = batch
        pred = model(x.to(device)).argmax(-1).cpu()
        cm += torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return cm


def metrics_from_confusion(cm):
    """Accuracy and macro precision/recall/F1 from accumulated counts."""
    cm = cm.double()
    tp = cm.diag()
    support = cm.sum(dim=1)    # true counts per class
    predicted = cm.sum(dim=0)  # predicted counts per class

    accuracy = (tp.sum() / cm.sum().clamp(min=1)).item()
    recall = tp / support.clamp(min=1)
    precision = tp / predicted.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "accuracy": accuracy,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class_recall": {str(c): round(r.item(), 4) for c, r in zip(CLASSES, recall)},
    }


def evaluate(model, loader, device):
    return metrics_from_confusion(confusion(model, loader, device))


def log_metrics(prefix, m):
    print(f"{prefix}: acc={m['accuracy']:.4f}  "
          f"macro P/R/F1={m['macro_precision']:.4f}/{m['macro_recall']:.4f}/{m['macro_f1']:.4f}  "
          f"recall per class={m['per_class_recall']}")


# --- training -----------------------------------------------------------------
def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    out_dir = (cfg_path.parent / str(cfg.out_dir)).resolve()
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    random.seed(cfg.data.seed)
    torch.manual_seed(cfg.data.seed)

    preprocess, color_aug = build_transforms(cfg)

    if cfg.data.get("file_list"):
        list_path = (cfg_path.parent / str(cfg.data.file_list)).resolve()
        paths = [Path(line) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"using cleaned file list: {list_path}")
    else:
        paths = sorted(p for p in Path(cfg.data.dir).rglob("*")
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert paths, "no images found"
    random.shuffle(paths)
    n_val = max(1, int(len(paths) * cfg.data.val_frac))
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    print(f"{len(train_paths)} train / {len(val_paths)} val pages")

    train_dl = DataLoader(TrainPages(train_paths, preprocess, color_aug),
                          batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, pin_memory=True,
                          collate_fn=collate_skip_broken)
    val_dl = DataLoader(ValPages(val_paths, preprocess),
                        batch_size=cfg.train.batch_size,
                        num_workers=cfg.train.num_workers, pin_memory=True,
                        collate_fn=collate_skip_broken)

    model = mobilenet_v3_small_page_orientation(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    sched = build_scheduler(opt, cfg.train.epochs * len(train_dl),
                            cfg.train.warmup_frac, cfg.train.min_lr_frac)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, best_f1 = 1, 0.0
    last_ckpt = out_dir / "last.pt"
    if cfg.train.get("resume", False) and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best_f1 = state["epoch"] + 1, state["best_f1"]
        print(f"resumed from last.pt: epoch {state['epoch']} done, best F1 {best_f1:.4f}")
    elif cfg.train.get("resume", False) and (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        print("resumed weights from best.pt (optimizer state reset, epoch counter restarts)")

    if start_epoch == 1:
        log_metrics("pretrained baseline", evaluate(model, val_dl, device))

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for batch in pbar:
            if batch is None:
                continue
            x, y = batch
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
            pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        print(f"epoch {epoch}: train loss={loss_sum / seen:.4f}")

        m = evaluate(model, val_dl, device)
        log_metrics(f"epoch {epoch} val", m)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  saved best.pt (macro F1 {best_f1:.4f})")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best_f1": best_f1}, last_ckpt)

    print(f"done: best macro F1={best_f1:.4f}, checkpoint saved to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
