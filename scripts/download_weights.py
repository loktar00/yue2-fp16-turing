#!/usr/bin/env python3
"""Download the YuE2 weights (about 7.3 GB) into the Hugging Face cache.

The repos are public and not gated, so no login or token is needed. Set HF_HOME
to put the cache somewhere other than ~/.cache/huggingface. The weights are
CC BY-NC 4.0, see MODEL_LICENSE.
"""
import argparse
import os

REPOS = ("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = ap.parse_args()
    print("cache:", os.environ.get("HF_HOME") or "~/.cache/huggingface (default)")
    for repo in REPOS:
        print("repo:", repo)
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download
    for repo in REPOS:
        path = snapshot_download(repo)
        print("ok:", repo, "->", path)


if __name__ == "__main__":
    main()
