---
name: research
description: Run one full research cycle in this lab. Takes a direction (or the top backlog idea), opens a study, checks novelty, derives and verifies the math, runs small GPU experiments against fair baselines, writes it up, gets it independently reviewed, and updates the index and backlog. Use when the user asks to research, investigate, explore an idea, find an improvement or make a discovery in AI math, architectures, solvers or training.
argument-hint: "[direction, backlog id like I-002, or study id to continue]"
---

# Research cycle

Direction from the user: $ARGUMENTS

If that is empty, take the highest-priority `ready` item in `ideas/BACKLOG.md`. If it names an existing study, continue that study instead of opening a new one.

`docs/PROTOCOL.md` is the authority; this is the checklist for applying it. Keep the task list current, one step per stage below.

## 0. Orient
- Read `STUDIES.md`, `ROADMAP.md`, `ideas/BACKLOG.md`, `docs/COMPUTE.md`, and the README of every study the direction builds on.
- Run `python3 tools/studies.py check`, and `nvidia-smi` to see whether this machine has a GPU. Without one, the cycle stops at E2 (E1 if you can't run Python), and the experiments go into the backlog for someone with a GPU.
- Note which claims you will rely on, with their evidence levels. Only claims at E2 or above can be built on.

## 1. Hypothesis
Write one falsifiable hypothesis: the claim, the measurable success criterion, the result that would refute it, and the evidence level the study aims for. Example: "The carried inverse with forget gates matches the token-by-token gated recurrence to 1e-12 in float64, and its float32 error after 65k tokens stays below 30 CG steps' error at lower cost." If you can't state it this sharply, narrow the idea first.

## 2. Open the study
```bash
python3 tools/studies.py new <slug> --title "<title>" --track <A-E>
```
Fill in `builds_on` and `tags`, write the hypothesis as Claim 1, and add the first NOTEBOOK entry. Mark the backlog item `taken: NNNN`.

## 3. Novelty check, before heavy work
Run `/novelty-check` (it uses the `literature-scout` subagent). If the method is already published, stop: write the study up as `refuted` (known), with what was learned and the reference, then move to the next idea, unless reproducing or extending it is worth doing and you say why.

## 4. Derive
Write the math in Method: definitions, assumptions, derivation, cost in FLOPs and memory, and where exactness or stability could fail.

## 5. Verify numerically (E2)
Write `check_<name>.py`: float64 on CPU against a direct reference (token-by-token loop, explicit inverse, autograd or finite differences), random inputs plus edge cases (ill-conditioned inputs, tiny and large sizes, lengths that don't divide the chunk size). Log the output to `logs/`. If it fails, fix the derivation or record the refutation before moving on.

## 6. Experiments (E3, E4)
Follow `/run-experiment`. Smoke test, then full runs. Use the strongest fair baseline, 3+ seeds for training claims, and record every run in NOTEBOOK.md, including failed and discarded ones.

## 7. Write up
Fill the README from the template: a Summary with numbers and sources, the Claims table with evidence levels, a calibrated verdict, specific limitations, next steps and reproduce commands. Set `status: review`, `evidence`, `summary` and `updated` in the front matter.

## 8. Independent review
Run `/review-study NNNN`. Fix every blocker and major issue by rerunning, correcting the text or narrowing the claim, and record each resolution in REVIEW.md. If blockers were found, review once more. Set the final status only when no blocker or major issue is open: `validated`, `refuted` or `inconclusive`. Otherwise leave `review` and list what is open.

## 9. Close out
- Promote reusable, validated code to `lab/` with a test (PROTOCOL section 10).
- Add follow-up ideas to `ideas/BACKLOG.md` and mark the taken item `done: NNNN`.
- Run `python3 tools/studies.py index`, then `python3 tools/studies.py check`.
- Set `authors` in the front matter: the GitHub handle of the person running the session and your model name.
- Report to the user: hypothesis, result with key numbers, verdict, evidence level, open issues and the suggested next step.
- Leave everything uncommitted unless the user asked for a commit or pull request; then commit on a new branch, push it, and open the pull request as `CONTRIBUTING.md` describes.

## Budget and when to ask
- By default, at most about 3 GPU hours per cycle and 60 minutes per run, with peak memory under 90% of the GPU. If the budget runs out, write up what you have as `inconclusive`, with the experiment that would decide it.
- Ask the user before a run longer than 60 minutes, a dataset download over 5 GB, installing new packages, or anything outside this repository.
