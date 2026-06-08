"""Download a large pool of free **handwritten Cyrillic** fonts into assets/fonts/.

Primary source: the ``fonts/`` folder of NastyBoget/HandwritingGeneration — a
curated, handwriting-only collection (~75 fonts with Cyrillic). We auto-discover
it via the GitHub API so you always get the current set; if the API is
unavailable (rate limit / offline) we fall back to a baked-in file list. A small
curated set of Google-Fonts handwriting families is added too.

These sources are handwriting/script-oriented, so printed (typeset-looking) fonts
are kept to a minimum. Coverage is verified later by ``FontBank`` (Latin-only or
uppercase-only fonts are dropped automatically), so it is fine to grab them all.

    python scripts/fetch_fonts.py
    python scripts/fetch_fonts.py --out /custom/dir --no-google
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

NASTY_REPO = "NastyBoget/HandwritingGeneration"
NASTY_BRANCH = "master"
NASTY_DIR = "fonts"
NASTY_RAW = f"https://raw.githubusercontent.com/{NASTY_REPO}/{NASTY_BRANCH}/{NASTY_DIR}/"

# Fallback list (used only if the GitHub API can't be reached).
_NASTY_FALLBACK = [f"{i:04d}.ttf" for i in range(10)] + [
    "Abram.ttf", "Anselmo.ttf", "BadScript-Regular.ttf", "Beer-Money.ttf", "Benvolio.ttf",
    "Capuletty.ttf", "Caveat-Regular.ttf", "Denistina.ttf", "Discipuli-Britannica.ttf",
    "Djiovanni.ttf", "Epsilon.ttf", "Eskal.ttf", "Example.ttf", "FestusC.ttf", "Gogol.ttf",
    "Gregory.ttf", "Gunnyre.ttf", "HansHand-cyr.ttf", "Katherine-Plus.ttf", "Lazy-Crazy.ttf",
    "Lorenco.ttf", "Marutya.ttf", "May-Regular.ttf", "Meamury.ttf", "Merkucio.ttf",
    "Montekky.ttf", "NinaC.ttf", "PFScandalPro-Reg.ttf", "Pag.ttf", "Paris.ttf", "Pushkin.ttf",
    "Salavat.ttf", "Samson.ttf", "Spring-Blush.ttf", "Stefano.ttf", "Tibalt.ttf", "VSerikba.ttf",
    "Vasek-Italic.ttf", "Voronov.ttf", "Wolgast-Two-Normal-Cyr.ttf",
    "Blink-Script.otf", "Brush-Font-One.otf", "Celestina.otf", "Elfabe.otf", "Hitch-hike.otf",
    "Lemon-Tuesday.otf", "MADE-Likes.otf", "Pinata-Celestina.otf", "Romochka.otf", "Simphony.otf",
    "Solena.otf", "StudioScriptC.otf", "Swanky-And-Moo-Moo-Cyrillic.otf", "Tesla.otf",
    "Tino-Script.otf",
]

# Curated Google-Fonts handwriting families known to ship Cyrillic.
_GOOGLE = [
    "ofl/marckscript/MarckScript-Regular.ttf",
    "ofl/badscript/BadScript-Regular.ttf",
    "ofl/neucha/Neucha.ttf",
    "ofl/pangolin/Pangolin-Regular.ttf",
    "ofl/pacifico/Pacifico-Regular.ttf",
    "ofl/caveat/Caveat[wght].ttf",
    "ofl/yesevaone/YesevaOne-Regular.ttf",
    "ofl/underdog/Underdog-Regular.ttf",
]
_GOOGLE_RAW = "https://raw.githubusercontent.com/google/fonts/main/"


def _github_listing(repo: str, branch: str, path: str) -> list[tuple[str, str]]:
    """(name, download_url) for font files in a repo dir, via the GitHub API. [] on failure."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    req = urllib.request.Request(url, headers={"User-Agent": "synth-fetch-fonts/1.0",
                                               "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return [(it["name"], it["download_url"]) for it in data
                if it.get("type") == "file" and it["name"].lower().endswith(FONT_EXTS)]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError):
        return []


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "synth-fetch-fonts/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
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
    ap.add_argument("--no-google", action="store_true", help="skip the Google-Fonts set")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # (filename, url) — dedup by filename across sources
    entries: dict[str, str] = {}

    listing = _github_listing(NASTY_REPO, NASTY_BRANCH, NASTY_DIR)
    if listing:
        print(f"NastyBoget: discovered {len(listing)} fonts via GitHub API")
        for name, dl in listing:
            entries.setdefault(name, dl)
    else:
        print(f"NastyBoget: API unavailable, using baked-in list ({len(_NASTY_FALLBACK)})")
        for name in _NASTY_FALLBACK:
            entries.setdefault(name, NASTY_RAW + urllib.parse.quote(name))

    if not args.no_google:
        for rel in _GOOGLE:
            entries.setdefault(rel.split("/")[-1], _GOOGLE_RAW + urllib.parse.quote(rel))

    print(f"Downloading up to {len(entries)} fonts -> {out}\n")
    ok = skip = fail = 0
    for name, url in sorted(entries.items()):
        dest = out / name
        if dest.exists():
            skip += 1
            continue
        if _download(url, dest):
            ok += 1
        else:
            print(f"  FAIL  {name}")
            fail += 1

    print(f"\nDone: {ok} downloaded, {skip} already present, {fail} failed.")
    n_fonts = sum(1 for p in out.iterdir() if p.suffix.lower() in FONT_EXTS)
    print(f"Fonts in {out}: {n_fonts}")
    if n_fonts == 0:
        print("\nNo fonts available. Grab handwriting+cyrillic .ttf from fontesk.com / "
              "localfonts.eu / fontspace.com and drop them into the dir above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
