# Contributing

Anyone with a Claude Code session can contribute research to this lab: theory from a phone, numerical checks from any computer, experiments from a GPU. Pick a prompt in [PROMPTS.md](PROMPTS.md), let Claude run it on your fork, and open a pull request. This page says how, and what a pull request needs to be merged.

## How to contribute

1. **[Fork this repository](https://github.com/MaurerKrisztian/automated-AI-research/fork)** on GitHub.
2. **Start a Claude Code session on your fork.**
   - On a phone or in a browser: open the Claude app or claude.ai/code and start a session on your fork. It runs in the cloud, with Python and web search but no GPU.
   - On your computer: clone your fork and run `claude` in it. Set up Python with `scripts/setup_env.sh` if you have an NVIDIA GPU, or `scripts/setup_env.sh --cpu` if you don't.
3. **Paste a prompt** from [PROMPTS.md](PROMPTS.md) and let it run. Answer its questions; it asks for your GitHub handle so the study credits you.
4. **Read the result** before opening the pull request. The paper's Summary and Claims should say exactly what was shown, and every number should cite a file in the study.
5. **Open a pull request** from your fork to this repository's `main`. If Claude can't open it against this repository, open it yourself on GitHub: on your fork's page, choose Contribute, then Open pull request.

## What fits your machine

| Where Claude runs | Setup | Reaches |
|---|---|---|
| The Claude app or claude.ai/code (cloud) | None; `pip install numpy scipy` for float64 checks | E1, and E2 with NumPy |
| A computer without an NVIDIA GPU | `scripts/setup_env.sh --cpu`, or only `pip install numpy scipy` | E2 |
| A computer with an NVIDIA GPU | `scripts/setup_env.sh`, then `.venv/bin/python scripts/gpu_probe.py` | E3–E4, and E5 when the GPU can train a 10M+ parameter language model in reasonable time |

On Apple silicon, run float64 checks on the CPU, because the GPU backend (MPS) has no float64, and don't compare MPS timings with CUDA ones.

## What a pull request contains

- **One unit of work:** one new study, one review, one novelty check, or one set of fixes to one study.
- **For a study:** `README.md` (the paper), `NOTEBOOK.md`, the `REVIEW.md` from your `/review-study` run, the scripts, and the logs and results behind every number.
- **A passing check:** `python3 tools/studies.py check` runs on every pull request.
- **Small files only:** no datasets, checkpoints or files over 5 MB. Datasets go in `data/`, which git ignores, and the study says where to get them.
- **Respect for other studies:** change someone else's study only through a dated erratum at the top of its README, or a review in its `REVIEW.md`.
- **A description:** what was done, the evidence level reached, the model that did it, and the hardware behind any timing.

## Study numbers

Take the next free number when you create a study; `python3 tools/studies.py new` does this. If another pull request takes the same number first, update your branch from `main` and run `python3 tools/studies.py renumber studies/NNNN-your-slug`, which moves your study to the next free number.

## Status and review

- New studies arrive with status `review`, or `refuted` or `inconclusive` when that's the result.
- Your own `/review-study` run is a self-check. A study becomes `validated` only after a review by a different contributor, merged in its own pull request, that leaves no blocker or major issue open.
- Negative results are welcome. A clean refutation, or the finding that an idea is already published, saves everyone time.

## Credit

List yourself and the model in the study's front matter, for example `authors: ["@your-handle", "Claude Opus 5.5"]`. The website shows each study's authors.

## Ground rules

The rules in [CLAUDE.md](CLAUDE.md) apply to every session: traceable numbers, immutable logs, fair baselines, 3+ seeds for training claims, calibrated wording, and no silent rewrites of earlier work. In reviews, criticize claims, not people.
