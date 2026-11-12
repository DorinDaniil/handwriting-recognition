"""Pages of a photo with the trained page model, before the line detector.

    cfg = OmegaConf.load("config.yaml")
    finder = PageFinder(cfg, "outputs/best.pt")
    for page in finder(photo):                         # pages in reading order, upright
        boxes = detector(page.image)                   # [n, 4, 2] corners (x, y) in page.image
        boxes = page.to_photo(boxes)                   # the same corners in the photo
    draw(photo, finder.quads(photo)).show()            # where the pages were found
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

from data import letterbox
from model import PageNet
from pages import Page, cut, find


class PageFinder:
    def __init__(self, cfg, ckpt, device="cpu"):
        self.cfg, self.device = cfg, device
        self.model = PageNet(cfg.model.width, pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model.to(device).eval()

    @torch.no_grad()
    def maps(self, photo: Image.Image):
        """-> (paper [h, w], core [h, w] probabilities, map pixels per photo pixel)"""
        x, scale = letterbox(photo, self.cfg.data.image_size)
        prob = torch.sigmoid(self.model(torch.from_numpy(x)[None].to(self.device)))[0].cpu().numpy()
        return prob[0], prob[1], scale / self.cfg.data.stride

    def quads(self, photo: Image.Image) -> list:
        """-> page quads in photo coordinates (TL, TR, BR, BL), in reading order"""
        return find(*self.maps(photo), photo.size)

    def __call__(self, photo: Image.Image, margin: float = 0.03) -> list[Page]:
        return [cut(photo, quad, margin) for quad in self.quads(photo)]


def draw(photo: Image.Image, quads) -> Image.Image:
    """A copy of the photo with every quad outlined and numbered."""
    out = photo.convert("RGB")
    pen = ImageDraw.Draw(out)
    width = max(2, round(max(out.size) / 300))
    for k, quad in enumerate(quads, 1):
        points = [tuple(p) for p in np.asarray(quad, np.float64)]
        pen.line(points + points[:1], fill=(255, 40, 40), width=width)
        pen.text(tuple(np.mean(points, 0)), str(k), fill=(255, 40, 40), anchor="mm", font_size=max(16, 12 * width))
    return out
