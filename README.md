# AI Research Lab

An open lab where AI agents research the mathematics of AI models. They propose ideas, check them against the literature, derive them, test them with small experiments, and write each one up as a short paper, failures included. Every study gets an independent review before it counts, and validated results become building blocks for later studies, with the long-term aim of combining them into better models.

## Claude usage left this week? Turn it into research

If your weekly limit resets soon and you still have usage left, don't let it go to waste: point Claude at this repository, let it run a research cycle, and open a pull request. A phone is enough for theory, reviews and literature checks; any computer can run numerical checks; a GPU can run experiments.

1. [Fork this repository](https://github.com/MaurerKrisztian/automated-AI-research/fork).
2. Start a Claude Code session on your fork: in the Claude app, at claude.ai/code, or with `claude` in a local clone.
3. Paste a prompt from [PROMPTS.md](PROMPTS.md), let it run, and open a pull request. [CONTRIBUTING.md](CONTRIBUTING.md) has the details.

| You have | You can | Prompt |
|---|---|---|
| A phone | Write a theory study (E1): an idea, its derivation and a literature check, as a short paper | [Theory study](PROMPTS.md#1-theory-study-e1) |
| A phone | Go for something big: brainstorm ideas, drop the published ones, develop the best | [Moonshot](PROMPTS.md#2-moonshot) |
| A phone | Review a study, or check an idea against prior work | [Review](PROMPTS.md#3-review-a-study), [Novelty check](PROMPTS.md#4-novelty-check) |
| Any computer, or a cloud session | Check a derivation numerically in float64 (E2) | [CPU verification](PROMPTS.md#5-cpu-verification-e2) |
| An NVIDIA GPU | Run benchmarks and train small models (E3–E4) | [GPU experiment](PROMPTS.md#7-gpu-experiment-e3e4) |

The Moonshot prompt works from a phone and goes as far as your setup allows:

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md, CONTRIBUTING.md and docs/PROTOCOL.md, then STUDIES.md, ROADMAP.md and ideas/BACKLOG.md.

Goal: find one idea that would make AI models meaningfully cheaper to train or run, or better at the same cost, if it holds.

1. Brainstorm 10 candidates across the roadmap's tracks. For each, give the mechanism in one or two sentences and why it might work.
2. Search the literature and drop the candidates that are already published. Tell me which survived and why.
3. Take the most promising survivor as far as this session can go, following /research: E1 (derivation) if you can't run code, E2 (float64 check on the CPU) if you can, E3 or E4 if nvidia-smi shows a GPU.
4. Be ambitious about the idea and strict about the claims: the write-up states exactly what you showed, at the evidence level it earned. A refuted idea is a fine result.
5. Put my GitHub handle (ask me if you can't find it) and your model name in the study's authors.
6. Add the other surviving candidates to ideas/BACKLOG.md so others can pick them up.
7. Run /review-study on the study and fix what it finds. Run python3 tools/studies.py index and check, then commit on a new branch, push it, and open a pull request titled "Study NNNN: <title> (E<level>)".
```

## How the lab works

Each study lives in `studies/NNNN-slug/`. Its `README.md` is the paper: method, math, results, limitations and references. `NOTEBOOK.md` logs everything tried, `REVIEW.md` holds the independent review, and `logs/` and `results/` hold the raw output behind every number.

Every claim carries an evidence level, so readers can see how far it has been tested:

| Level | Meaning |
|---|---|
| E0 | Stated: an idea, nothing checked |
| E1 | Derived: the derivation or proof is written out |
| E2 | Verified: matches a direct reference in float64 |
| E3 | Benchmarked: measured against a fair baseline |
| E4 | Small-model evidence: synthetic tasks, 3+ seeds |
| E5 | Language-model evidence: tiny or small language models against a tuned baseline |

Later studies build only on claims at E2 or above, and validated code moves into the shared `lab/` package. The rules every session follows are in [CLAUDE.md](CLAUDE.md), the full protocol is in [docs/PROTOCOL.md](docs/PROTOCOL.md), and the plan is in [ROADMAP.md](ROADMAP.md).

## Studies

[STUDIES.md](STUDIES.md) lists every study with its status and evidence level. The website at [maurerkrisztian.github.io/automated-AI-research](https://maurerkrisztian.github.io/automated-AI-research/) publishes the papers, reviews, notebooks and results with full-text search, and the prompts with copy buttons. It's built from `site/` on every push to `main`.

## Running it yourself

```bash
scripts/setup_env.sh
```

That creates `.venv` with CUDA PyTorch (about 3 GB to download). On a machine without an NVIDIA GPU, add `--cpu`. Then:

| Command | What it does |
|---|---|
| `python3 tools/studies.py check` | Validates every study |
| `.venv/bin/python -m unittest discover -s tests` | Tests the shared code in `lab/` |
| `cd site && npm install && npm run dev` | Previews the website |

## Layout

```
README.md            this page
PROMPTS.md           ready-to-paste prompts for contributors
CONTRIBUTING.md      how to contribute, and what a pull request needs
CLAUDE.md            rules every agent session follows
STUDIES.md           generated index of studies
ROADMAP.md           goal, research tracks and milestones
ideas/BACKLOG.md     ranked queue of testable ideas
docs/                research protocol, compute budget, GPU probe log
studies/NNNN-slug/   one study: paper, notebook, review, code, logs, results
lab/                 shared Python package: benchmarking, results, validated layers
tests/               tests for lab/
tools/studies.py     create, check, index, renumber and export studies
templates/study/     starting files for a new study
scripts/             environment setup and GPU probe
site/                the website (React + Vite, GitHub Pages)
.claude/             skills, subagents and settings for Claude Code
```
