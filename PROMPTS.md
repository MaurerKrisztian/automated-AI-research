# Prompts

Copy a prompt, paste it into a Claude Code session on your fork of this repository, and let it run. Claude Code loads the lab's rules from `CLAUDE.md` by itself; the prompts point it at the rest.

## Where to run them

| Where | Setup | Highest evidence level |
|---|---|---|
| The Claude app on your phone, or claude.ai/code in a browser | Start a Claude Code session on your fork. It runs in the cloud with Python and web search, but no GPU. | E2 |
| Your computer without an NVIDIA GPU | Clone your fork and run `claude` in it. Run `scripts/setup_env.sh --cpu` if you need PyTorch. | E2 |
| Your computer with an NVIDIA GPU | Clone your fork, run `scripts/setup_env.sh`, then `claude`. | E3–E5 |

Evidence levels: E1 means derived on paper, E2 checked numerically in float64, E3 benchmarked, E4 small models trained, E5 language models trained. [docs/PROTOCOL.md](docs/PROTOCOL.md) has the full definitions.

## Pick a prompt

| Prompt | Needs | Session | Produces |
|---|---|---|---|
| [1. Theory study](#1-theory-study-e1) | A phone is enough | Medium | A new study at E1 |
| [2. Moonshot](#2-moonshot) | A phone is enough; goes further with a GPU | Long | A new study, plus ideas for the backlog |
| [3. Review a study](#3-review-a-study) | A phone is enough | Short | A review |
| [4. Novelty check](#4-novelty-check) | A phone is enough | Short | A prior-work search for one idea |
| [5. CPU verification](#5-cpu-verification-e2) | Python, cloud sessions included | Medium | A claim moved from E1 to E2 |
| [6. Fix a study's review issues](#6-fix-a-studys-review-issues) | Depends on the issues | Medium | Resolved review issues |
| [7. GPU experiment](#7-gpu-experiment-e3e4) | An NVIDIA GPU | Long | A study at E3 or E4 |

"Open a pull request" in a prompt means one from your fork to this repository. [CONTRIBUTING.md](CONTRIBUTING.md) says what it needs.

## 1. Theory study (E1)

No code: reading, web search and writing. Produces a new study written as a short paper.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md, CONTRIBUTING.md and docs/PROTOCOL.md, then STUDIES.md and ideas/BACKLOG.md.

Do one theory-only research cycle that aims for evidence level E1. Don't run experiments.

1. Pick a ready backlog item marked "Runs on: phone", or propose your own idea in the mathematics of AI models: one precise claim that would make models cheaper or better if it holds.
2. Search the literature before deriving anything. If the idea is already published, write it up as a study with status refuted and the verdict "Known", cite the paper, and stop there.
3. Create the study with python3 tools/studies.py new <slug> --title "<title>" --track <A-E>.
4. Write its README.md as a short paper: definitions, the claim stated as a proposition, a complete derivation or proof, the cost in FLOPs and memory, where it could fail, and an exact plan for the E2 numerical check. Set evidence: E1 and status: review.
5. Put my GitHub handle (ask me if you can't find it) and your model name in authors.
6. Run /review-study on the study and fix what the review finds.
7. Add a backlog item for the E2 check, then run python3 tools/studies.py index and python3 tools/studies.py check.
8. Commit on a new branch, push it, and open a pull request titled "Study NNNN: <title> (E1)".
```

## 2. Moonshot

Aims for an idea that would matter if it holds, then takes it as far as your setup allows: E1 on a phone, E2 with Python, E3 or E4 with a GPU.

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

## 3. Review a study

Checks someone else's study against its own logs. Reviews by a different contributor are what move studies to `validated`.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md and docs/PROTOCOL.md (sections 5 to 8), then STUDIES.md.

Review one study: the one I name, otherwise a study with status "review" whose REVIEW.md is missing or oldest.

1. Run /review-study on it.
2. Confirm every issue yourself against the study's logs/, results/ and code before keeping it. Drop what you can't confirm.
3. Write only REVIEW.md, keeping earlier reviews under "Earlier reviews". Don't change the study's paper.
4. Name the reviewer as my GitHub handle (ask me if you can't find it) plus your model name.
5. Commit on a new branch, push it, and open a pull request titled "Review of NNNN".
```

## 4. Novelty check

Searches prior work for one backlog idea, so nobody spends days rediscovering a published result.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md, docs/PROTOCOL.md (section 7) and ideas/BACKLOG.md.

Check one backlog idea against prior work: the item I name, otherwise the highest-priority item without a novelty check.

1. Run /novelty-check on it.
2. Under the backlog item, add a "Novelty" line with the date, the closest works (title, year, link) and the verdict: "already published as …", "no prior publication found as of <date>", or "a classical result in a new setting".
3. If the idea is already published, set the item's status to "dropped (published as …)", unless reproducing it is still worth doing; then say why.
4. Commit on a new branch, push it, and open a pull request titled "Novelty check: I-NNN".
```

## 5. CPU verification (E2)

Checks a derivation numerically in float64. Works in cloud sessions too: it needs Python, not a GPU.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md, CONTRIBUTING.md and docs/PROTOCOL.md, then STUDIES.md and ideas/BACKLOG.md.

Bring one claim from E1 to E2 by checking it numerically in float64 on the CPU. No GPU and no training.

1. Pick a study claim at E1, or a ready backlog item marked "Runs on: CPU".
2. Install only what you need: pip install numpy scipy, or scripts/setup_env.sh --cpu if you need PyTorch.
3. Write check_<name>.py in the study's folder. Compare against a direct reference (token-by-token loop, explicit inverse, autograd or finite differences) on random inputs and edge cases: ill-conditioned inputs, tiny and large sizes. Keep the run to a few minutes.
4. Run it from the study folder with python -u check_<name>.py 2>&1 | tee logs/check_<name>.log, and cite that log in the paper's Claims table.
5. A failed check is a result too: record it. If the study is someone else's, add a dated erratum at the top of its README instead of rewriting their text.
6. Update the summary, Claims and evidence level, add a NOTEBOOK.md entry, then run /review-study and fix what it finds.
7. Run python3 tools/studies.py index and check, then commit on a new branch, push it, and open a pull request titled "Study NNNN: E2 check".
```

## 6. Fix a study's review issues

Works on any setup: each session takes the issues it can handle. Study 0001 has text-only issues that need no GPU.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md and docs/PROTOCOL.md, then the README.md, NOTEBOOK.md and REVIEW.md of the study I name, or of study 0001 if I don't name one.

Work through the open issues in its REVIEW.md that this session can handle: text fixes and literature from anywhere, float64 checks with Python, new runs only on an NVIDIA GPU.

1. Fix each issue you take on. New runs go in new log files; never edit old logs or results.
2. Set each fixed issue's Resolution to "fixed: …", naming the file that shows it. Leave the rest open and say why.
3. If results changed, update the summary and verdict in the same edit, and add a NOTEBOOK.md entry.
4. Run python3 tools/studies.py index and check, then commit on a new branch, push it, and open a pull request titled "Study NNNN: fix review issues".
```

## 7. GPU experiment (E3–E4)

For machines with an NVIDIA GPU. Benchmarks against fair baselines, or trains small models on synthetic tasks.

```text
You're contributing to an open AI research lab in this repository.
Read CLAUDE.md, CONTRIBUTING.md, docs/PROTOCOL.md and docs/COMPUTE.md, then STUDIES.md and ideas/BACKLOG.md.

This machine has an NVIDIA GPU. Run one research cycle with experiments, aiming for E3 (benchmarks against fair baselines) or E4 (small models on synthetic tasks, 3+ seeds).

1. Set up with scripts/setup_env.sh, run .venv/bin/python scripts/gpu_probe.py, and tell me which GPU you found and how much memory it has.
2. Pick a ready backlog item marked "Runs on: GPU", or a study claim at E2 that needs benchmarking, and run /research on it.
3. Size every run to this GPU: smoke test first, stay under 90% of its memory, keep each run under an hour unless I approve more, and run long jobs in the background with their logs in logs/.
4. Use the strongest fair baselines with an equal tuning budget, and compare speeds only between runs on this machine.
5. Put my GitHub handle (ask me if you can't find it) and your model name in the study's authors. Run /review-study and fix every blocker and major issue.
6. Run python3 tools/studies.py index and check, then commit on a new branch, push it, and open a pull request titled "Study NNNN: <title> (E3)", or E4.
```

## Shortcuts

In a Claude Code session on this repository, these commands do the same jobs with less typing. They stop before committing; when you're happy with the result, ask Claude to "commit on a new branch and open a pull request".

| Command | What it does |
|---|---|
| `/research` | A full research cycle on the top backlog idea |
| `/research I-002` | A full research cycle on one backlog item |
| `/review-study 0001` | Reviews study 0001 |
| `/novelty-check <idea>` | Searches prior work for an idea |
