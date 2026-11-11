---
title: Telling a project story: constraint, decision, number, limit
summary: How I prepare each project on my résumé to be told in thirty seconds or ten minutes: one sentence, five words to keep the order, a number that says where it came from, and the limits said before anyone asks.
group: Method
order: 1
tags: résumé · interviews · communication
---
A project story usually breaks on the second question, not the first. "I cut generation compute by about 42%" survives the opening; it breaks at "how did you measure that?" if the honest answer is "I didn't, it follows from the schedule" and the speaker has not decided to say so. Every story in this book was rebuilt around that moment, so each number carries its origin and each claim carries its limit.

This page is the method the other pages follow. It is one person's habit, written down; I have no data on whether it changes outcomes.

## One sentence first

Each project starts as one sentence with three parts: the constraint, what I did about it, and the result with its unit.

- "With only 500 labeled CT volumes, I started from my own hemorrhage model and reached 0.88 AUC."
- "At a recall floor set by policy, I raised precision by about 25 points, mostly with hard negatives."

If the sentence will not come, the story is not ready, and adding detail will not fix it. The constraint comes first because it explains every later decision: the label budget explains the model size, the recall floor explains the metric.

## Five words to keep the order

Under pressure the order goes before the facts do. So each story gets a chain of five words, one per step, each causing the next:

| Story | Chain |
|---|---|
| Best-of-N pruning | eight candidates · most discarded · score early · drop half · 14 of 24 |
| CT triage | 500 labels · 2.5D · own pretraining · cross-slice context · 0.88 |
| Smoking detector | human review · small objects · two imbalances · hard negatives · +25 points |

The chain is for recall, not for saying aloud. If I can say why each word leads to the next, I can rebuild the story from it.

## The order of the telling

Every story here follows the same five beats, and each has a job:

1. **Context.** Where, what system, who used the output. One or two sentences.
2. **Constraint.** The thing I did not choose: a latency budget, a label budget, a recall floor. This is the beat people skip and the one that makes the decisions make sense.
3. **Decision.** Two or three choices, each with its reason and its cost. "I used a detector because the objects were small" is a decision; "I used YOLOX" is a fact.
4. **Result.** The number, with its unit, its operating point and how it was obtained.
5. **Reflection.** What I would check or change today. This is where the known gaps go, said by me first.

I keep two lengths. The thirty-second version is context, the core decision and the number. The ninety-second version adds the constraint, all the decisions and the reflection. Anything longer is an answer to a question, not a monologue.

## Every number says where it came from

Most broken stories are a number that sounds measured and was not. I sort every number into one of three kinds before I use it:

| Kind | Example from this book | How to say it |
|---|---|---|
| Measured on a held-out set | 0.88 AUC on the test split | "on the held-out test set" |
| Derived from the design | 42% less compute: 14 candidate segments instead of 24 | "follows from the schedule" |
| Measured on a reproduction | about 1.2 ms per caption segment, timed on my rebuild | "on my reproduction, not the client's system" |

A number that fits none of these, such as an impression from a meeting, does not go on the résumé. I removed two that way: an efficiency gain nobody timed, and a speed-up larger than the schedule's own ceiling, which meant the baseline differed in some way I could not isolate.

A quick check catches many inconsistencies. If precision went from 55% to 80% at a fixed recall, the false positives must fall by about two thirds; the arithmetic is on the [detector's page](../precision-at-a-recall-floor/). If the arithmetic disagrees with the claim, one of them is wrong.

## Say who did what

"We" hides ownership, and the follow-up questions will find it. For each project I draw the system and mark each part as mine or someone else's: in the captions project, I designed the correction layer; the streaming integration, the caption display and deployment belonged to the platform team. Then I can say "I designed the algorithm; the platform team deployed it" and go as deep as the questions want on my part, and stop cleanly at the edge of theirs.

## Limits, said first

Each story has a short table of what I can say and what I should not:

| Can say | Should not say |
|---|---|
| It stayed a research prototype | It was deployed clinically |
| No noticeable drop in a side-by-side comparison | No significant drop (no test was run) |
| The main gain came from hard negatives | How much each step contributed (no ablation) |

Saying a limit before it is asked changes the conversation from catching a gap to discussing how to close it. So the reflection beat names the biggest gap and the experiment that would close it: for pruning, logging early and final scores on a few hundred unpruned runs and checking how often the eventual best is dropped.

## The two likeliest follow-ups

For each story I write down the two questions most likely to come next and answer them aloud. They are almost always "how do you know?" about the number and "why not the obvious alternative?" about the main decision. The honest answer often starts with "I didn't measure that", followed by how I would.

## Practice

Fifteen minutes a session:

1. Record the thirty-second version and time it; it should stay under forty seconds.
2. Pick three follow-up questions at random and answer each aloud.
3. Listen back for three things only: does each number say where it came from, did I state the limit unprompted, and did I say "we" where it was "I"?
4. Rewrite the sentence that stuck into words I would actually say.

## The thread

Across projects I look for one sentence that ties them together, so the stories read as one career rather than a list. In my case the common thread is a limit someone else set, which the model had to live within: reviewers who expect a minimum catch rate, a few hundred labeled scans, a caption that must appear within 50 ms. Each project page names its constraint in its first paragraph for that reason.

## What this does not settle

This method makes a story checkable; it does not make the work better than it was. It also leans on memory: where I could not recall a detail, such as the size of a test set, the page says so instead of guessing. What I have not tested is whether listeners prefer the limits early or late, and whether a shorter chain of three words works as well as five.
