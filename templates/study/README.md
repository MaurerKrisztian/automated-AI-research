---
id: "{id}"
title: {title}
status: active
evidence: E0
track: {track}
builds_on: []
tags: []
authors: []
summary: Open question: what this study tests, in one sentence.
updated: {date}
---

# {title}

## Summary

What was tried, the main result with its numbers and sources, and what it means. Keep this section current: when a result changes, change it in the same edit.

**Verdict:** not decided yet.

## Claims

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | The hypothesis, stated so a result can refute it | E0 | – |

## Problem

What is being improved, why it matters, and the current best approach.

## Method

Definitions, assumptions, the derivation, and the cost in FLOPs and memory. If this builds on earlier studies, name each claim used and its evidence level. Display math goes in `math` fences:

```math
x_t = (\lambda I + H_t)^{-1} q_t
```

## Results

Each table or figure with its baseline, precision, sizes and source file.

## Novelty check

Searched on YYYY-MM-DD: sources and queries (the full log is in NOTEBOOK.md).

| Work | What it does | Difference |
|---|---|---|

## Limitations

## Next steps

## References

Every work cited above, with a link (arXiv, DOI or venue page). Only works you opened.

1. Authors, "Title", venue year. https://arxiv.org/abs/…

## Reproduce

Run from this folder with the repository's virtual environment:

```bash
../../.venv/bin/python -u check_identity.py 2>&1 | tee logs/check_identity.log
```
