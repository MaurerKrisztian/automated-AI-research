---
name: review-study
description: Independent adversarial review of a study. Launches the read-only reviewer subagent, checks its findings, writes REVIEW.md, and says what must change before the study can be validated. Use after a write-up is drafted, before setting a study's status to validated, or when the user asks to review or check a study.
argument-hint: "<study id, e.g. 0001>"
---

# Review a study

Study: $ARGUMENTS

1. Find the folder `studies/<id>-*/` and run `python3 tools/studies.py check`.
2. Launch the `reviewer` subagent with the study folder's path and today's date. Ask it to apply the checklist in `docs/PROTOCOL.md` section 8, to verify every number in the Summary and Claims against `logs/` and `results/` itself, and to return a verdict, a ranked issue table and a list of what holds up.
3. Check the reviewer's blocker and major issues yourself by opening the cited files. Drop any that are wrong, and record that you dropped them and why.
4. Write `REVIEW.md` in this format. If the file holds an earlier review, move it under "Earlier reviews" at the bottom; never delete one.

   ```
   # Review of NNNN

   Reviewer: <GitHub handle of the person running the session> with <subagent and model> · Date: YYYY-MM-DD · Reviewed: README as of <its updated date>

   ## Verdict
   **Accept | Revise | Reject.** Two or three sentences.

   ## Issues
   | # | Severity | Issue | Evidence | Resolution |
   |---|---|---|---|---|
   | 1 | blocker / major / minor | What is wrong, quantitatively | file, line or computation | open |

   ## What holds up
   - Claims verified against their sources.

   ## Earlier reviews
   ```

5. Don't fix the issues during the review. Fixing belongs to the study's research cycle; each fix then updates the Resolution column (`fixed: …` or `won't fix: …`).
6. If any blocker or major issue is open, the study's status must be `review`.
7. Report the verdict and the top issues to the user.
