# YuE2 for Turing GPUs (fp16)

A runnable setup of [YuE2](https://github.com/multimodal-art-projection/YuE), the open lyrics-to-song model, patched to run in **float16** so it works on **RTX 20-series (Turing, sm_75)** cards, which have no bfloat16 and no FlashAttention 2. It includes a small web UI and HTTP API: type a style and lyrics, get a full stereo song back as FLAC.

Upstream hard-codes bfloat16. This repo changes that to a `YUE2_DTYPE` environment switch that defaults to `float16`. The only other change makes the CUDA-graph decoder fall back to plain SDPA on pre-Ampere cards.

- Tested on: **RTX 2080 Ti 22 GB (modded), fp16**. Same-seed A/B against bf16 on an RTX 5090: fp16 generated cleanly and both sounded good.
- Peak VRAM: about **7.8 GiB** in fp16 (measured on the 5090 run, so it should be the same on a 2080 Ti; the AR model is parked on the CPU while the VAE decodes). A 22 GB card has a lot of headroom.
- A stock **11 GB 2080 Ti**: untested. It should fit on paper (upstream has a low-memory path), see [VRAM notes](#vram-notes).
- Speed: about 0.3x real time on a 5090 with FlashAttention (a 45 s song in ~16 s). Turing cannot use the fast attention kernels, so this repo selects plain SDPA there; forcing that path on the 5090 took ~1x real time. Not measured on a 2080 Ti; expect a few minutes per song.

## Credits and licenses

This repo is a fork-style redistribution. All the real work is by the YuE2 authors.

- **YuE2** by the YuE2 authors / M-A-P: <https://github.com/multimodal-art-projection/YuE> (demo: <https://map-yue2.github.io/>). The inference code in `src/yue2/` is vendored from upstream commit `92a73cc` (v0.1.6, 2026-09-10) under the **Apache License 2.0**; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (Stability AI stable-audio-tools, MIT; NVIDIA BigVGAN SnakeBeta, MIT; texts in [licenses/](licenses/)). Upstream copyright notices are kept.
- **Model weights** (`m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`) are **not in this repo**; you download them from Hugging Face. They are licensed **CC BY-NC 4.0 plus an individual creator permission** (upstream `MODEL_LICENSE`, dated 2026-09-16; a copy of the current text is in [MODEL_LICENSE](MODEL_LICENSE)). In plain words, as upstream summarises it:
  - Personal users, content creators and musicians acting as individuals may use YuE2 free of charge and may publish, sell or otherwise monetize the songs they generate, with no fees or royalties.
  - Academic research and education: free for non-commercial use.
  - Companies using the weights commercially need a commercial license from the YuE2 authors (contact details in the upstream README).
  - Responsible use is a condition: nothing illegal, harmful or deceptive (fraud, harassment, impersonating a real person). Provided as is, no warranty.
  - Credit on generated songs is optional but encouraged ("YuE2" or #YuE2).
  - Read `MODEL_LICENSE` yourself; this summary is not legal advice.
- The fp16 patch, the web UI/API service and the install scripts in this repo are also Apache 2.0.

What changed from upstream is listed in [PATCHES.md](PATCHES.md).

## Requirements

- Linux (or Windows via WSL2 with the NVIDIA driver installed on the Windows side), Python 3.10 or newer (3.11 is what was tested), `git`.
- NVIDIA GPU. For Turing use `YUE2_DTYPE=float16` (the default here). Driver 570+ for the default `cu128` torch wheel; use `--cuda cu126` for older drivers.
- About 8 GB of free disk for the weights, plus the Python environment (~5 GB).

## Quick start

```bash
git clone https://github.com/loktar00/yue2-fp16-turing.git
cd yue2-fp16-turing

scripts/install.sh            # venv + torch 2.10 (CUDA wheel) + yue2 + service deps + weights (~7.3 GB)
scripts/run.sh                # starts the service on http://127.0.0.1:5830
```

Open <http://127.0.0.1:5830/>, adjust the style and lyrics (or tick Instrumental) and press Generate. The first song also loads the model, so it takes longer. Songs are saved under `outputs/<UTC time>-<name>/` as `audio.flac` (plus `audio.wav`, the request and the score).

Install options: `scripts/install.sh --help` (`--cuda cu126|cu128|cu129|cu130`, `--python python3.11`, `--no-weights`, `--dry-run`). There is no FlashAttention, xformers or vLLM to install: YuE2 uses PyTorch's built-in SDPA attention, which runs on Turing.

Weights are public and **not gated**: no Hugging Face login or token is needed. To store them elsewhere, set `HF_HOME=/some/dir` before installing and before running. To fetch them later: `.venv/bin/python scripts/download_weights.py`.

Check that torch sees your card: `.venv/bin/python scripts/check_gpu.py`. It should print `sm_75` for a 2080 Ti and recommend float16.

### Command line / API

```bash
.venv/bin/python scripts/make_song.py --request examples/city_lights.json
.venv/bin/python scripts/make_song.py --request examples/forge_instrumental.json
.venv/bin/python scripts/make_song.py --id mysong --style "English, ..., 90 BPM" --lyrics-file my_lyrics.txt --seeds 3

curl -s localhost:5830/health
curl -s -X POST localhost:5830/v1/generate -H 'Content-Type: application/json' \
  -d '{"id":"test","style":"English, warm piano pop, female voice, 88 BPM","lyrics":"[Verse]\nHello world\n\n[Chorus]\nSing it all night"}'
```

Endpoints: `GET /` UI, `GET /health`, `POST /load`, `POST /unload`, `POST /v1/generate` (`style`, `lyrics` or `instrumental:true`, `seed`, `cot`, `id`, `cfg_scale`), `GET /v1/generate/{job}/audio?fmt=flac|wav`, `GET /v1/jobs`.

The service binds to `127.0.0.1`. `YUE2_HOST=0.0.0.0 scripts/run.sh` exposes it to your network; it has no login, so only do that on a trusted LAN.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `YUE2_DTYPE` | `float16` | `float16`, `bfloat16` (Ampere and newer only) or `float32` |
| `YUE2_OUT_DIR` | `./outputs` | where songs are written |
| `YUE2_IDLE_UNLOAD_SECONDS` | `600` | free the GPU after this much idle time (`0` = keep loaded) |
| `YUE2_GPU` | `0` | which CUDA device to use |
| `YUE2_MEMORY_BUDGET_GIB` | card size | VRAM budget handed to the pipeline |
| `YUE2_OFFLOAD_AR` | `0` | `1` keeps the AR model on the CPU while decoding (low VRAM) |
| `YUE2_ATTENTION` | auto | CUDA-graph attention kernel: `sdpa` (auto-selected on cards older than sm_80), `flash`, `cudnn` |
| `YUE2_BACKEND` | `torch` | `torch-eager` disables CUDA graphs if they misbehave |
| `YUE2_VERIFY_HASHES` | `1` | `0` skips the checksum pass over the weights on each load |
| `YUE2_HOST`, `YUE2_PORT` | `127.0.0.1`, `5830` | listen address (used by `run.sh`) |

## Writing good prompts

Every song is a **style** sentence plus a **lyrics** field. There are no tag lists or genre presets.

- **Style is one free-text sentence, in this order: language, genre, instruments, vocal character, BPM.** Not a pile of keywords.
  > English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM
- **Lyrics** use `[Verse]`, `[Chorus]`, `[Bridge]` tags with a blank line between sections. Two to four sections is plenty.
- **Any word in the lyrics field gets sung, including words inside brackets.** A line like `[Intro: music box alone, no vocals]` will often get sung. Put direction in the style sentence, not the lyrics.
- **Instrumentals:** the lyrics field must be bare `[Instrumental]` tags and nothing else (the Instrumental checkbox / `"instrumental": true` does exactly that). Drop the language and vocal character from the style and spend the sentence on mood, instruments, key and BPM; ending it with "no vocals" helps.
  > Dark orchestral forge theme, low strings, anvil hits, taiko drums, brass swells, minor key, 115 BPM, no vocals
- **Length is a lottery.** The number of sections does not control duration: identical instrumental requests came out at 36, 108, 259 and 294 seconds across four seeds. If length matters, try several seeds. A take that hits the ~360 s ceiling is flagged `truncated` and ends abruptly.
- The same seed and the same request reproduce the same song (within one dtype), so note the seed of takes you like.
- `Planning` (`cot`): `full` writes a chord-annotated score first (default), `melody` is melody only, `off` is fastest.

## VRAM notes

- fp16 weights are about 6.8 GB for the 3B model plus a 0.5 GB VAE. Measured peak in fp16: about 7.8 GiB (RTX 5090 run).
- **22 GB 2080 Ti:** plenty of room; the default settings are meant for this card.
- **11 GB 2080 Ti: untested.** Upstream's pipeline has a low-memory mode that kicks in at a budget of 12 GiB or less (smaller VAE chunks) and an `offload_ar` option. Try `YUE2_MEMORY_BUDGET_GIB=11 YUE2_OFFLOAD_AR=1 scripts/run.sh`. If it works for you, please open an issue and say so.
- The first load needs about 9.5 GB of system RAM.

## Troubleshooting

- **Silent, noisy or black output, or NaN errors.** fp16 has a smaller range than bf16. First re-run with another seed. If it keeps happening, try `YUE2_DTYPE=float32 scripts/run.sh`: weights then take about 13 GB, so it needs a 22 GB card, and it is slower. (This switch applies to the whole model, not to individual stages; the VAE already runs in float32.)
- **`CUDA is not available`.** The NVIDIA driver is missing or too old for the wheel. Run `nvidia-smi`; try `scripts/install.sh --cuda cu126`, or update the driver. On WSL2, install the driver on Windows, not inside WSL.
- **`no kernel image is available for execution`.** Your torch build lacks sm_75. Run `scripts/check_gpu.py`; reinstall with a `cu126`/`cu128` wheel.
- **CUDA out of memory.** Close other GPU programs (browser hardware acceleration can hold GBs), make sure nothing else is loaded (`nvidia-smi`), then try `YUE2_OFFLOAD_AR=1` and `YUE2_BACKEND=torch-eager`. Songs that run close to the length ceiling use the most memory.
- **Slow first run.** The first request downloads the weights (if you skipped it at install) and hashes them (~25 s from a cold disk; `YUE2_VERIFY_HASHES=0` skips it).
- **The page says `unloaded`.** That is the normal resting state: the model is loaded by the first request and freed after the idle timeout.

## Repo layout

```
src/yue2/            vendored YuE2 inference package (fp16 patch: dtype.py, pipeline.py, fast.py, cuda_graph.py)
service/             serve_yue2.py (FastAPI service) and index.html (web UI)
scripts/             install.sh, run.sh, download_weights.py, check_gpu.py, make_song.py
examples/            example request JSON, lyrics, and an instrumental request
patches/             the fp16 patch as a diff against upstream
LICENSE, MODEL_LICENSE, THIRD_PARTY_NOTICES.md, licenses/   upstream license files
```

## Not included

Covers, SheetSage2 melody transcription, editing, the upstream agent skills and the optional vLLM "fast" backend are upstream features that are not wired into this UI (the vLLM path is patched for dtype but untested). For those, use the upstream repo.
