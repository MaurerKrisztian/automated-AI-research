# AI research lab: agent guide

This repository is an open research lab for the mathematics of AI models: layers, solvers, optimizers and training methods. You propose ideas, validate them with small experiments, and document them so later studies can build on them. The long-term aim is to combine validated components into a better model. Honest, reproducible results matter more than impressive ones, and a clean negative result counts as a success.

You may be working for the maintainer or for an outside contributor, in a cloud session on a phone, on a laptop, or on a GPU workstation. Fit the work to the session: with no way to run code, aim for E1 (a derivation); with Python, E2 (float64 checks); with an NVIDIA GPU, E3 and up. `CONTRIBUTING.md` says what a pull request needs, and `PROMPTS.md` holds the prompts contributors paste.

## Start here
- `STUDIES.md`: every study with its status, evidence level and key result. Read it before starting work.
- `ROADMAP.md`: the goal, research tracks and milestones. `ideas/BACKLOG.md`: ranked queue of testable ideas, each marked with what it runs on.
- `docs/PROTOCOL.md`: the full research protocol (stages, evidence levels, review checklist). Read it before your first study in a session.
- `docs/COMPUTE.md`: GPU limits on the reference machine, and how to size work on other machines.

## Layout
- `studies/NNNN-slug/`: one study each. `README.md` is the study's paper (method, math, results; shown as the Paper tab on the website), with front matter. `NOTEBOOK.md` is a dated log of everything tried, `REVIEW.md` the independent review. Experiment scripts sit at the top level; `logs/` and `results/` hold immutable outputs; `archive/` holds superseded material.
- `lab/`: shared Python package. `lab.bench` (timing, memory, errors, seeding), `lab.results` (save JSON with environment info), `lab.layers` (validated components promoted from studies).
- `tests/`: unittest checks for `lab/`. `tools/studies.py`: `new`, `check`, `index`, `renumber`, `export`. `templates/study/`: files a new study starts from.
- `site/`: the searchable website (React + Vite) built from the studies; see `site/README.md`.
- `PROMPTS.md`, `CONTRIBUTING.md`: the contributor prompts and the pull-request guide.

## Environment
- Python: `.venv/bin/python` after `scripts/setup_env.sh` (CUDA PyTorch) or `scripts/setup_env.sh --cpu` (CPU PyTorch). For float64 checks alone, `pip install numpy scipy` is enough. Once `setup_env.sh` has run, `lab` is importable from any folder.
- Cloud sessions (Claude Code on the web or in the Claude app) have Python and web search but no GPU.
- GPU: the reference machine is an RTX 5060 Ti with 16 GB (`docs/COMPUTE.md`). On any GPU, run `nvidia-smi` before a job and run one GPU job at a time. On another machine, run `scripts/gpu_probe.py` first and compare speeds only between runs on the same machine.
- Study scripts run from their study folder: `cd studies/NNNN-slug && ../../.venv/bin/python -u script.py ... 2>&1 | tee logs/<name>.log`.
- Tests: `.venv/bin/python -m unittest discover -s tests`. Study checks: `python3 tools/studies.py check`.
- Website: `cd site && npm run dev` to preview, `npm run build` to build.

## Skills and agents
- `/research [direction or backlog id]`: one full research cycle, from idea to reviewed write-up.
- `/novelty-check <idea>`: prior-work search before investing in an idea.
- `/run-experiment`: conventions for GPU runs, logs and results.
- `/review-study NNNN`: independent adversarial review, saved to `REVIEW.md`.
- Subagents: `reviewer` (read-only referee) and `literature-scout` (prior-work search).

## Rules
1. **Traceable numbers.** Every number in a write-up comes from a file in that study's `logs/` or `results/`, and the write-up cites the file. Never write a number from memory or from a run that left no file.
2. **Immutable records.** Never edit or delete files in `logs/` or `results/`. Rerun and add new files, and mark superseded runs in `NOTEBOOK.md`.
3. **Fair baselines.** Compare against the strongest version of the baseline you can run: its published settings, an equal tuning budget, and its real backward pass or kernel where one exists. Say plainly what was not optimized.
4. **Seeds and spread.** Training claims need at least 3 seeds per configuration, reported as mean and min–max. A difference inside the seed spread is not a result.
5. **Math first.** Verify every identity and gradient in float64 against a direct reference (token-by-token loop, autograd or brute force) before benchmarking it. A claim you could only derive stays at E1.
6. **Current summary.** A README's summary and verdict must match its latest results. When a result changes, update the summary in the same edit.
7. **Calibrated claims.** Don't write "breakthrough", "novel" or "state of the art". Use the evidence levels (E0–E5) and verdict wording from the protocol, and write "no prior publication found as of <date>" only after a documented search.
8. **Negative results count.** Write up a refuted idea with status `refuted` and what was learned.
9. **Build on evidence.** Cite the study id and evidence level of each claim you rely on, and rely only on claims at E2 or above. If an earlier study turns out wrong, add an erratum to its README and change its status. Never silently rewrite it, and never rewrite another contributor's text.
10. **Fit the machine.** Smoke-test at small size first. Keep peak GPU memory under 90% of the card (14 GB on the reference GPU). Keep a single run under 60 minutes unless the person running the session approved more, and run anything over about 2 minutes in the background with its log in `logs/`.
11. **Git.** Work on a branch, never directly on `main`. Commit, push or open a pull request only when the person running the session asks for it; the contributor prompts do. Never force-push.
