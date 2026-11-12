# Page layout

Finds the pages on a photo and cuts each one out upright, before the handwriting line detector.
A notebook photo is often a spread with two pages, lies at an angle and shows the table around it;
the detector works better on one clean page at a time, and its lines never run across the binding.

## Two maps

`PageNet` (MobileNetV3-small encoder, small top-down decoder, ~1M parameters) takes the photo
letterboxed into 512 x 512 and predicts two maps at 1/4 resolution:

- `paper` — every pixel of any page;
- `core` — each page with its border peeled off, so two pages that touch on `paper` still form
  two separate blobs.

`pages.py` turns them into pages: every `core` blob is a page, each `paper` pixel goes to the
nearest core, the outline gives four corners, and the quad is warped into an upright crop.
No page found means the whole photo is the page.

## Data

- ai-forever School Notebooks (RU and EN), COCO category `paper`: one polygon per page, many spreads.
  `data.notebooks` takes HuggingFace dataset ids (read from the local cache) or local snapshot folders.
- Optional `data.scans`: folders of single-page scans that fill the frame; samples put one sheet,
  or two side by side, on a table.

## Training

    python train.py [config.yaml]

Checkpoints go to `outputs/` (`best.pt`, `last.pt`; training resumes from `last.pt`). Validation is
the test split; the score per photo is the IoU of matched pages divided by the larger page count.
At start the log prints `photos by number of pages`: it must show photos with 2 pages, otherwise
the spreads were not read and the model never learns to split them.

## Usage

    from omegaconf import OmegaConf
    from infer import PageFinder, draw

    finder = PageFinder(OmegaConf.load("config.yaml"), "outputs/best.pt")
    for page in finder(photo):                  # PIL image -> pages in reading order
        boxes = detector(page.image)
        boxes = page.to_photo(boxes)            # back to photo coordinates
    draw(photo, finder.quads(photo))            # the photo with the pages outlined

`test.ipynb` shows training samples with both maps and the trained model on your own photos.
