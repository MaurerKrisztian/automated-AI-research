# Roadmap

## Goal

Find mathematical improvements to AI models that survive fair tests, then combine them into a model that beats a tuned baseline at equal compute, at the largest scale this GPU allows (see [docs/COMPUTE.md](docs/COMPUTE.md)).

## How results compound

1. A study validates a component at E2–E4: an exact identity, a faster solver, a better memory layer.
2. The component moves into `lab/` with tests.
3. Later studies import it, combine it with other components, and test the combination one evidence level higher.

Every step cites the claims it relies on (study id and evidence level), so a later failure can be traced back to its source.

## Tracks

| Track | Question | Studies |
|---|---|---|
| A. Sequence memory and attention | Can fixed-size memories (linear attention, delta rule, least-squares memory) approach softmax attention's recall at lower cost? | 0001 |
| B. Solvers and numerics | Exact or better-conditioned algebra for layers that now iterate or approximate, and stability in low precision | 0001 (partly) |
| C. Optimization | Optimizers and preconditioners that reach the same loss with fewer FLOPs at small scale | – |
| D. Parameterization and stability | Initialization, normalization and scaling rules that transfer across width and depth | – |
| E. Theory probes | Capacity and expressivity results, and synthetic tasks that predict language-model quality | – |

## Milestones

| # | Milestone | State |
|---|---|---|
| M1 | A component validated at E4 with fair baselines | In progress: 0001 is at E3, with partial E4 evidence under review |
| M2 | A shared tiny-LM baseline in `lab/`: tuned Transformer, data pipeline and training loop, 3 seeds (backlog I-005) | Not started; needed before any E5 claim |
| M3 | A validated component matches or beats the tiny-LM baseline at equal parameters and tokens | Not started |
| M4 | Two or more validated components combined, with ablations showing each one helps | Not started |
