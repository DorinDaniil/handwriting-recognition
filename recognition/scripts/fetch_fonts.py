"""Bootstrap free handwriting fonts: RU (Cyrillic) -> assets/fonts_ru, EN (Latin) -> assets/fonts_en.

A starter set only — point the config at your own font folders if you have them.

    python scripts/fetch_fonts.py
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FONT_EXTS = (".ttf", ".otf", ".ttc")

NASTY = ("NastyBoget/HandwritingGeneration", "master", "fonts")
NASTY_RAW = f"https://raw.githubusercontent.com/{NASTY[0]}/{NASTY[1]}/{NASTY[2]}/"
_NASTY_FALLBACK = [f"{i:04d}.ttf" for i in range(10)] + [
    "Abram.ttf", "Anselmo.ttf", "BadScript-Regular.ttf", "Benvolio.ttf", "Capuletty.ttf",
    "Caveat-Regular.ttf", "Denistina.ttf", "Djiovanni.ttf", "Eskal.ttf", "Gogol.ttf",
    "Gunnyre.ttf", "HansHand-cyr.ttf", "Marutya.ttf", "Merkucio.ttf", "NinaC.ttf",
    "Pushkin.ttf", "Salavat.ttf", "Voronov.ttf", "Wolgast-Two-Normal-Cyr.ttf",
    "Celestina.otf", "Romochka.otf", "Solena.otf", "Swanky-And-Moo-Moo-Cyrillic.otf",
]

_GOOGLE_EN = [
    "ofl/caveat/Caveat[wght].ttf", "ofl/dancingscript/DancingScript[wght].ttf",
    "ofl/indieflower/IndieFlower-Regular.ttf", "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "ofl/patrickhand/PatrickHand-Regular.ttf", "ofl/kalam/Kalam-Regular.ttf",
    "ofl/pacifico/Pacifico-Regular.ttf", "ofl/sacramento/Sacramento-Regular.ttf",
]
GOOGLE_RAW = "https://raw.githubusercontent.com/google/fonts/main/"


def _github_listing(repo, branch, path):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-fonts/1.0",
                                               "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return [(it["name"], it["download_url"]) for it in data
                if it.get("type") == "file" and it["name"].lower().endswith(FONT_EXTS)]
    except Exception:
        return []


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-fonts/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if len(data) < 1024:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def _populate(entries, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for name, url in sorted(entries.items()):
        dest = out / name
        if dest.exists():
            skip += 1
        elif _download(url, dest):
            ok += 1
        else:
            print(f"  FAIL {name}"); fail += 1
    n = sum(1 for p in out.iterdir() if p.suffix.lower() in FONT_EXTS)
    print(f"  {out}: +{ok} new, {skip} present, {fail} failed -> {n} fonts")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1] / "assets"
    ap.add_argument("--ru-out", type=Path, default=root / "fonts_ru")
    ap.add_argument("--en-out", type=Path, default=root / "fonts_en")
    args = ap.parse_args()

    print("RU (Cyrillic):")
    listing = _github_listing(*NASTY)
    ru = ({n: u for n, u in listing} if listing
          else {n: NASTY_RAW + urllib.parse.quote(n) for n in _NASTY_FALLBACK})
    _populate(ru, args.ru_out)

    print("EN (Latin, Google Fonts):")
    _populate({rel.split("/")[-1]: GOOGLE_RAW + urllib.parse.quote(rel) for rel in _GOOGLE_EN}, args.en_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
