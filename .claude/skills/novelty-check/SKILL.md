---
name: novelty-check
description: Search papers, preprints and code for prior work on an idea before investing in it, and record the search in the study. Use before starting a study, whenever a write-up says an idea is new, or when the user asks whether something has been done before.
argument-hint: "<idea, or a study id>"
---

# Novelty check

Idea or study: $ARGUMENTS

If this is a study id, read that study's README to get the idea.

1. **State the idea precisely.** Write it as one to three technical statements: the math, not the pitch. Then list 5–10 phrasings an author might use, including the names other fields use for the same math (for example recursive least squares, Kalman filter, Woodbury update, fast weights, test-time regression, online learning).
2. **Search.** Launch the `literature-scout` subagent with the statements, the phrasings and today's date. For a broad topic, split it into two or three scouts with different angles (ML literature, other fields, code) and run them in parallel.
3. **Judge overlap.** Sort the closest works into: same method, same math in another field, partial overlap, or related only. Open each candidate yourself before you rely on it: the scout can misjudge overlap.
4. **Record.** In the study's NOTEBOOK.md: date, queries, sources searched and hits. In the README's Novelty check: a table of Work, What it does, Difference, followed by the conclusion.
5. **Conclude with calibrated wording**, one of:
   - "Already published as …" (then the study's verdict is "Known").
   - "No prior publication found as of YYYY-MM-DD (searched: …). Closest: …"
   - "A classical result (…) applied in a new setting."

Never write "novel" or "first". A search can't rule out unpublished or concurrent work, so say so.
