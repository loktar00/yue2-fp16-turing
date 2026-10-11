#!/usr/bin/env python3
"""Make a song through a running service (standard library only).

  python scripts/make_song.py --id city_lights --style "English, warm piano pop, ..." --lyrics-file examples/city_lights.lyrics.txt
  python scripts/make_song.py --id forge --instrumental --style "Dark orchestral forge theme, ... 115 BPM"
  python scripts/make_song.py --request examples/city_lights.json

Downloads each take's FLAC into --out-dir as <id>-seed<seed>.flac.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        sys.exit("service said %d: %s" % (exc.code, exc.read().decode(errors="replace")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("YUE2_URL", "http://127.0.0.1:5830"))
    ap.add_argument("--request", help="JSON file with id/style/lyrics/seed/cot (command-line flags override it)")
    ap.add_argument("--id")
    ap.add_argument("--style")
    ap.add_argument("--lyrics")
    ap.add_argument("--lyrics-file")
    ap.add_argument("--instrumental", action="store_true")
    ap.add_argument("--sections", type=int, default=4, help="number of [Instrumental] tags; does not control length")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, default=1, help="number of takes at seed, seed+1, ...")
    ap.add_argument("--cot", choices=("full", "melody", "off"))
    ap.add_argument("--out-dir", default="outputs/takes")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    req = {}
    if a.request:
        with open(a.request, encoding="utf-8") as fh:
            req = json.load(fh)
    if a.id:
        req["id"] = a.id
    if a.style:
        req["style"] = a.style
    if a.lyrics_file:
        with open(a.lyrics_file, encoding="utf-8") as fh:
            req["lyrics"] = fh.read()
    if a.lyrics:
        req["lyrics"] = a.lyrics
    if a.instrumental:
        req["instrumental"] = True
        req["sections"] = a.sections
    if a.cot:
        req["cot"] = a.cot
    base_seed = a.seed if a.seed is not None else req.get("seed", 831001)
    if not req.get("style"):
        ap.error("a style is required")
    req.setdefault("id", "song")

    os.makedirs(a.out_dir, exist_ok=True)
    for i in range(a.seeds):
        req["seed"] = base_seed + i
        print("generating %s seed %d ..." % (req["id"], req["seed"]), flush=True)
        out = post(a.url + "/v1/generate", req, a.timeout)
        dest = os.path.join(a.out_dir, "%s-seed%d.flac" % (req["id"], req["seed"]))
        urllib.request.urlretrieve(a.url + out["flac_url"], dest)
        print("  %s  %.1f s of audio in %.1f s, peak %.2f GiB, truncated=%s"
              % (dest, out["duration_s"], out["seconds"], out["peak_vram_gib"], out["truncated"]))


if __name__ == "__main__":
    main()
