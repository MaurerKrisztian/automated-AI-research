# Research protocol

How every study in this lab is run, judged and written up. `CLAUDE.md` has the short version of the rules; this file has the detail. Where a skill and this protocol disagree, the protocol wins.

## 1. The research cycle

| Stage | What you do | Output | Done when |
|---|---|---|---|
| 0. Orient | Read `STUDIES.md`, `ROADMAP.md`, `ideas/BACKLOG.md` and every study the idea builds on | – | You know what is already established |
| 1. Hypothesis | State one falsifiable claim, its success criterion, and the result that would refute it | NOTEBOOK entry | The cheapest decisive experiment is clear |
| 2. Novelty | Search for prior work (`/novelty-check`) | Novelty section, search log | No identical published method, or a stated reason to reproduce it anyway |
| 3. Derivation | Write the math: definitions, assumptions, derivation or proof, cost | Method section | Every step can be checked |
| 4. Verification (E2) | Check the math in float64 against a direct reference | `check_*.py`, its log | Errors at float64 round-off (about 1e-12 or better), or the idea is refuted |
| 5. Benchmark (E3) | Speed, memory or accuracy against the strongest fair baseline | logs, results | Numbers are stable across reruns |
| 6. Small models (E4/E5) | Train small models, 3+ seeds, baselines tuned with the same budget | logs, results | Differences exceed the seed spread, or clearly don't |
| 7. Write-up | Fill `README.md` from the template | README | Every number cites its file |
| 8. Review | Independent reviewer agent (`/review-study`) | `REVIEW.md` | No open blocker or major issue |
| 9. Close-out | Index, backlog, promotion to `lab/` | `STUDIES.md`, backlog, `lab/` | Section 12 is done |

Stop as soon as a stage refutes the idea, and write it up anyway with status `refuted`, so no later agent repeats it. Not every study needs every stage: a pure-math study can end at E2 and a numerics study at E3. The hypothesis says which level the study aims for.

## 2. Evidence levels

Every claim carries one level. A study's front-matter `evidence` is the level of its headline claim.

| Level | Name | Meaning |
|---|---|---|
| E0 | Stated | Idea and hypothesis written down; nothing checked |
| E1 | Derived | Derivation or proof written in the study, assumptions explicit |
| E2 | Verified | Matches a direct reference in float64 on random inputs and edge cases (identities, gradients, invariants) |
| E3 | Benchmarked | Speed, memory or accuracy measured on this GPU against a fair baseline, with logs |
| E4 | Small-model evidence | Small models trained on synthetic or toy tasks, 3+ seeds, baselines tuned with the same budget |
| E5 | Language-model evidence | A tiny or small language model (10M+ parameters, 100M+ tokens) against a tuned baseline at equal parameters and tokens, 3+ seeds at the smallest size |

Later studies may build only on claims at E2 or above, and must say which level they rely on. A claim never borrows a level it hasn't earned: an E3 speedup says nothing about model quality, and E4 recall on a synthetic task says nothing about language modeling.

## 3. Study status

| Status | Meaning |
|---|---|
| `active` | Work in progress |
| `review` | Written up; the review is pending or has open blocker or major issues |
| `validated` | Claims hold at their stated levels and the review has no open blocker or major issue |
| `refuted` | The central claim failed, or the method was already published |
| `inconclusive` | The budget ran out before a decision; the README says what would decide it |
| `superseded` | Replaced by a later study, named in `superseded_by` |

## 4. Study folder

```
studies/NNNN-slug/
  README.md      the paper: method, math and results; current and curated, with front matter
  NOTEBOOK.md    dated, append-only log of everything run or tried, including dead ends
  REVIEW.md      latest independent review, with the resolution of each issue
  *.py, *.sh     experiment code, run from this folder
  logs/          stdout of every run the write-up cites (immutable)
  results/       machine-readable outputs, JSON or Markdown (immutable)
  archive/       superseded material kept for the record
```

Create a study with `python3 tools/studies.py new <slug> --title "<title>" --track <A-E>`. It takes the next free number.

### Front matter

```
---
id: "0002"
title: Short descriptive title
status: active
evidence: E0
track: A
builds_on: ["0001"]
tags: [mesa, forgetting]
authors: ["@your-handle", "Claude Opus 5.5"]
summary: One sentence with the key result and its main number, or the open question.
updated: 2026-10-29
---
```

One `key: value` per line, lists in brackets. The format is kept this simple on purpose: `tools/studies.py` parses it without a YAML library, and `check` rejects anything else. `authors` (the contributor's GitHub handle and the model) and `superseded_by: "NNNN"` are optional.

If another study took your number first, `python3 tools/studies.py renumber studies/NNNN-slug` moves yours to the next free number.

## 5. Write-up

Each study's `README.md` is its paper: one Markdown file with the method, the math and the results, written to be read on its own. GitHub shows it when someone opens the study folder, and the website shows it as the study's Paper tab. `NOTEBOOK.md` and `REVIEW.md` support it but never replace it.

`README.md` sections, in order: Summary (ending with a **Verdict** line), Claims, Problem, Method, Results, Novelty check, Limitations, Next steps, References, Reproduce. `tools/studies.py check` requires Summary, Method and Results once a study is written up, and warns when References is missing.

- **Lead with the result.** The Summary says what was tried, the main numbers and the verdict. A reader who stops there must not be misled.
- **Cite every number.** Put the source after the number or under the table: `(logs/bench.log)`. The Claims table lists each claim with its evidence level and source.
- **Show the baseline next to every result**, at the same precision, sizes and hardware.
- **Write plainly.** Define each symbol where it first appears. Write display math in ```` ```math ```` fences (they render on GitHub and on the site) and inline math as `$...$` or Unicode.
- **Make limitations specific**: which sizes, tasks and baselines were not tested, and what could change the conclusion.
- **List references** with links (arXiv, DOI or venue page), only for works you actually opened.
- **Use calibrated verdicts**:

| Verdict | Use when |
|---|---|
| Refuted: … | The central claim failed |
| Known: published as … | The method already exists |
| Holds at small scale: … | Supported at E3/E4, not yet tested on language models |
| Promising: … | Supported at E4/E5 with a clear next test |
| Inconclusive: … | Not decided within the budget |

Avoid "breakthrough", "revolutionary", "novel", "state of the art" and "solves".

## 6. Fair baselines and statistics

**Baselines**
- Use the strongest version you can run: the authors' code or kernels when available (for example flash-linear-attention), their recommended hyperparameters, and the same precision.
- Give every method the same tuning budget. If yours gets a 4-value learning-rate sweep, so does each baseline.
- Hold everything else equal: data, steps, parameter count (state it), sequence length, precision and hardware.
- Count the real backward pass. If the baseline's paper uses a cheaper implicit or analytical backward, implement it, or say the comparison leaves it out and estimate its effect.
- Report the sizes where your method is weakest, not only where it shines.

**Timing**
- Use `lab.bench` (warmup, CUDA synchronization, median of repeats). Keep TF32 off unless TF32 is the subject, and state precision and sizes.
- Plain-PyTorch timings are pessimistic for every method; ratios are what matter, and fused kernels can change them. Say so.

**Statistics**
- Training claims need at least 3 seeds per configuration. Report the mean and min–max, or all values.
- Report every configuration you ran, including failures. Never pick the best seed.
- Call a difference inside the seed spread "no measurable difference".
- When a model fails to learn (loss near chance), say it did not train and diagnose why (learning rate, initialization, regularization) before blaming capacity.

**Numerics**
- Check exactness claims in float64 against a direct reference, then measure float32 error against float64.
- Test stability claims on badly conditioned inputs and long sequences, and report how the error grows, not only its final value.

## 7. Novelty check

- Run it before the derivation and experiments (stage 2) with the `literature-scout` agent or `/novelty-check`.
- Search arXiv, Semantic Scholar, OpenReview, conference proceedings and GitHub, under the names other fields use for the same math (for example recursive least squares, Kalman filtering, fast weights, test-time regression).
- Log the queries, the date and the closest works in `NOTEBOOK.md`. Summarize them in the README's Novelty check as a table: Work, What it does, Difference.
- Write "no prior publication found as of YYYY-MM-DD", followed by what was searched. Never write "novel" or "first".
- Discuss the closest techniques even when they are not the same method. A reviewer will ask about them.

## 8. Review

A study can't be `validated` until it has had an independent review. `/review-study NNNN` runs the `reviewer` agent on this checklist:

1. The Summary and Claims match the latest logs and results (no stale numbers), and every number has a source.
2. Baselines are the strongest feasible versions, with an equal tuning budget and their real backward passes and kernels.
3. Speed claims use the same precision for every method, realistic sizes, warmup and synchronization, and include the sizes where the method is weakest.
4. "Exact", "stable" and "no drift" claims hold over the whole tested range, worst cases included.
5. Training claims have 3+ seeds, differences larger than the spread, and every run reported.
6. Failures are diagnosed correctly (did not train, capacity limit, or bug), with loss curves checked.
7. The novelty search is documented and the closest techniques are discussed.
8. Conclusions don't extrapolate beyond the tested settings (sizes, head dimensions, tasks).
9. Limitations are stated and the verdict uses calibrated wording.
10. The README's commands reproduce the results from a clean checkout, with fixed seeds.

`REVIEW.md` holds a verdict (accept, revise or reject) and a table of issues. Severity levels:
- **blocker**: a central claim is wrong or unsupported.
- **major**: a claim is overstated, a baseline is unfair, or the evidence is too thin for the stated level.
- **minor**: clarity or a missing detail.

Each issue's resolution is `open`, `fixed` (how, and which file shows it) or `won't fix` (why). Keep earlier reviews at the bottom of `REVIEW.md` under "Earlier reviews".

A review run in the same session that wrote the study is a self-check. For contributed studies, `validated` takes a review by a different contributor, merged in its own pull request (see `CONTRIBUTING.md`).

## 9. Building on earlier studies

- List the studies in `builds_on`, and say in Method which claims you use and at what level.
- Import validated code from `lab/` instead of copying it out of another study's folder.
- If you find an earlier study wrong, add a dated **Erratum** at the top of its README, change its status, add a NOTEBOOK entry, and tell the user. Never rewrite history silently.

## 10. Promotion to `lab/`

Code moves from a study into `lab/` when:
- its claim is at E2 or above;
- it has a unittest in `tests/` against a direct reference (float64 on CPU, plus a GPU test when relevant);
- its docstring names the study and the evidence.

The study keeps its own copy. Studies are frozen records, while `lab/` may evolve.

## 11. Compute

See [COMPUTE.md](COMPUTE.md). Smoke-test first, keep peak GPU memory under 90% of the card (14 GB on the reference GPU), run one GPU job at a time, and run long jobs in the background. A single run stays under 60 minutes and a cycle under about 3 GPU hours unless the person running the session approves more. Without a GPU, stop at the highest level the session can reach (E1 without Python, E2 with it) and add the next experiment to the backlog.

## 12. End of a session

- Add a dated `NOTEBOOK.md` entry for what happened.
- Bring the README's Summary, Claims, status and `updated` date up to date.
- Run `python3 tools/studies.py index`, then `python3 tools/studies.py check`.
- Update the backlog: add follow-ups and mark taken items with the study id.
- Report to the user: the hypothesis, the result, the verdict, the evidence level, open issues and the next step.
- Leave changes uncommitted unless the person running the session asked for a commit or pull request. Then commit on a new branch and open the pull request as `CONTRIBUTING.md` describes.
