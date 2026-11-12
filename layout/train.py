"""Training of the page model on notebook photos with outlined pages (and optional scans).

Validation is the test split, framed as at inference. For every photo the predicted pages are
matched to the annotated ones; the score of a photo is the summed IoU of the matched pairs
divided by the larger of the two page counts. A spread read as one page, a page split in two
and a sloppy outline all lower it. The best checkpoint has the highest mean score.

Usage: python train.py [path/to/config.yaml]
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Pages
from model import PageNet
from pages import split


def loss_fn(logits, target):
    """BCE per pixel + Dice per map (Dice does not care how small the pages are)"""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum((2, 3)) + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


def build_scheduler(opt, total_steps, warmup_frac, min_lr_frac):
    """Per-step LR schedule: linear warmup, then cosine decay to lr * min_lr_frac."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def photo_score(found, n_found, ids):
    """found: predicted page labels, ids: annotated page ids, both [h, w] -> (score in [0, 1], same count)"""
    true = [ids == k for k in range(1, int(ids.max()) + 1) if (ids == k).any()]
    pred = [found == k for k in range(1, n_found + 1)]
    if not true or not pred:
        return float(not true and not pred), len(true) == len(pred)
    iou = np.array([[(t & p).sum() / (t | p).sum() for p in pred] for t in true])
    total = 0.0
    for _ in range(min(iou.shape)):                              # best pairs first, each page used once
        t, p = np.unravel_index(iou.argmax(), iou.shape)
        total += iou[t, p]
        iou[t, :], iou[:, p] = -1, -1
    return total / max(len(true), len(pred)), len(true) == len(pred)


@torch.no_grad()
def evaluate(model, loader, device):
    """mean photo score and the share of photos with the right number of pages"""
    model.eval()
    scores, counts = [], []
    for x, _, ids in tqdm(loader, desc="val", leave=False):
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
        for maps, true in zip(prob, ids.numpy()):
            score, same = photo_score(*split(maps[0], maps[1]), true)
            scores.append(score)
            counts.append(same)
    return float(np.mean(scores)), float(np.mean(counts))


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    out_dir = (cfg_path.parent / str(cfg.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    t = cfg.train
    train_dl = DataLoader(Pages(cfg, train=True), batch_size=t.batch_size, shuffle=True, drop_last=True,
                          num_workers=t.num_workers, pin_memory=True)
    val_dl = DataLoader(Pages(cfg, train=False), batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=True)

    model = PageNet(cfg.model.width, cfg.model.pretrained).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr)
    sched = build_scheduler(opt, t.epochs * len(train_dl), t.warmup_frac, t.min_lr_frac)

    start_epoch, best = 1, -1.0
    last_ckpt = out_dir / "last.pt"
    if t.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"resumed from last.pt: epoch {state['epoch']} done, best score {best:.4f}")

    for epoch in range(start_epoch, t.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for x, y, _ in pbar:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(x)
            seen += len(x)
            pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        score, count = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: train loss={loss_sum / seen:.4f}  val score={score:.4f}  right page count={count:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  saved best.pt (score {best:.4f})")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best}, last_ckpt)

    print(f"done: best score={best:.4f}, checkpoint saved to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
