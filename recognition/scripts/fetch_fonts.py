"""Download a starter set of free Cyrillic handwriting fonts into assets/fonts/.

All fonts below are OFL-licensed and ship with Cyrillic coverage (most are by
Russian/Cyreal foundries). Files are pulled directly from the google/fonts repo.
Anything that fails (404, network) is skipped — re-run any time; existing files
are not re-downloaded. You can also just drop your own .ttf/.otf into assets/fonts/.

    python scripts/fetch_fonts.py
    python scripts/fetch_fonts.py --out /custom/font/dir
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/google/fonts/main/"

# (display name, repo path under google/fonts). Coverage is verified later by FontBank.
FONTS: list[tuple[str, str]] = [
    ("Marck Script", "ofl/marckscript/MarckScript-Regular.ttf"),
    ("Bad Script", "ofl/badscript/BadScript-Regular.ttf"),
    ("Neucha", "ofl/neucha/Neucha.ttf"),
    ("Pangolin", "ofl/pangolin/Pangolin-Regular.ttf"),
    ("Pacifico", "ofl/pacifico/Pacifico-Regular.ttf"),
    ("Caveat", "ofl/caveat/Caveat[wght].ttf"),
    ("Lobster", "ofl/lobster/Lobster-Regular.ttf"),
    ("Ruslan Display", "ofl/ruslandisplay/RuslanDisplay.ttf"),
    ("Yeseva One", "ofl/yesevaone/YesevaOne-Regular.ttf"),
    ("Underdog", "ofl/underdog/Underdog-Regular.ttf"),
]


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "synth-fetch-fonts/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1024:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    ap.add_argument("--out", type=Path, default=default_out, help="font output dir")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    for name, rel in FONTS:
        fname = rel.split("/")[-1]
        dest = out / fname
        if dest.exists():
            print(f"  skip  {name:16s} ({fname})")
            skip += 1
            continue
        url = RAW + urllib.parse.quote(rel)
        if _download(url, dest):
            print(f"  ok    {name:16s} -> {fname}")
            ok += 1
        else:
            print(f"  FAIL  {name:16s} ({rel})")
            fail += 1

    print(f"\nDone: {ok} downloaded, {skip} already present, {fail} failed.")
    print(f"Fonts dir: {out}")
    if ok + skip == 0:
        print("\nNo fonts available. Download manually from fontesk.com / localfonts.eu "
              "(filter: handwriting + cyrillic) and drop .ttf files into the dir above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
