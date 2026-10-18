# Baby’s RNC curriculum progress

This is a short guide to the saved results from Sessions 001–007 on Fan Diesel. The lessons are small, generated curriculum panels, not a broad benchmark. Scores describe those panels only.

| Session | Focus | Saved result |
| --- | --- | --- |
| 001 | Chapter 1 pilot | Taught the answer format and clean turn endings; Chapter 1 dev accuracy was about 58%, leaving object binding weak. |
| 002 | Chapter 1 binding | Dev accuracy reached 63.0%, but missed the 70% target; multi-object questions and novel-scene accuracy also missed their targets. |
| 003 | Chapter 1 contrastive binding | Forced-choice binding rose from 46.4% to 91.6%; generated Chapter 1 dev rose from 63.0% to 88.3%. Chapter 4 dev fell from 26.0% to 20.6%. |
| 004 | Chapter 2 event memory | Chapter 2 forced choice rose from 44.4% to 95.6%; generated dev reached 76.3%. Chapter 4 recovered to 24.8%. Open/closed-state retention weakened. |
| 005 | Chapter 2 missing details | On dev_c2x, generated why/when reached 97.9% and open/closed forced choice 75.6%. Chapter 3 dev rose from 47.3% to 62.3%. Chapter 1 two-sentence descriptions regressed and missed their target. |
| 006 | Chapter 3 grounded answers | Full dev Chapter 3 rose from 62.3% to 91.0%. Chapter 1 binding forced choice finished at 83.1% against an 85% target; answerable Chapter 1–2 over-caution was 3.59% against a 3% limit. Full dev Chapter 4 was 25.8%. |
| 007 | Chapter 4 short conversations | Completed 100 updates and 3,215,658 input tokens. On the trainer’s sampled in-session dev turns, Chapter 4 rose from 18/86 (20.9%) at step 0 to 83/86 (96.5%) at update 100; all 86 final replies ended with EOS. On the same small panel, Chapters 1–3 were 49/64, 58/64, and 61/64 at step 0, then 51/64, 58/64, and 61/64. Chapter 2 dev_c2x was 221/256 (86.3%). |

Sessions used 100 optimizer updates each. The final Session 007 event records a normal completion at update 100, no stop condition, and an unchanged Session 006 parent. Its checkpoint SHA-256 is dc0c60766460c57db52f1bfcc0ab58c449b3a36fda3bee66db8c7861efe4832a. See [the frozen ancestor record](SESSION_007_COMMON_ANCESTOR.md).

## What Baby can and cannot claim

Saved evidence supports better performance on these generated tasks: binding details to the right object, retaining a sequence of events, answering some why/when questions, identifying ambiguity or missing facts, and short Chapter 4 dialogue turns. Earlier sessions also found weaknesses: comparisons, open/closed toggles, concise two-sentence descriptions, and retention tradeoffs. The known Session 006 binding and over-caution misses were not retested by Session 007’s in-session panel.

Chapter 4 evaluation generates each Baby reply with the reference earlier turns supplied as context. Its 86 checked turns are a slice of the 300-dialogue dev set, and no separate full-dev evaluation was run after Session 007. The score is therefore an encouraging result on a limited sample, not proof of end-to-end dialogue consistency. These curriculum results do not establish broad factual knowledge, general reasoning, durable memory, coding skill, or an independently trained specialist.

The underlying checkpoints, lesson arrays, raw evaluation rows, and private workspaces remain local. This repository contains this summary and checkpoint identity records only. S1/S2/S3 refer to preserved MRCN-Alpha history; they are not active RNC gates.
