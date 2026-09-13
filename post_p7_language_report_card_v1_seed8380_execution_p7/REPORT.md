# P7 post-P7 language report card execution

This is the single authorized evaluation of the immutable P7 archive on the frozen battery. No training occurred; the frozen battery was not modified. Naturalistic rubric fields await human review.

## Controlled results

### near_distribution
Items: 96; item accuracy: 0.500000; ties: 0; complete families: 0/12; mean within-family reversal success: 0.062500; margin mean/min/median/max: -0.061399/-7.621537/-0.060103/7.761543

Deterministic representative sample:

PROMPT:
Ben found the blue ball. Sam carried the blue book.
Who found the blue ball?

CORRECT CANDIDATE:  Ben.
COMPETING CANDIDATE:  Sam.
CANDIDATE LOG-LIKELIHOODS: [-22.556595355693386, -16.627666512123476]
MARGIN: -5.92892884356991
SELECTED ANSWER:  Sam.
RESULT: incorrect

### counterfactual
Items: 64; item accuracy: 0.500000; ties: 0; complete families: 0/8; mean within-family reversal success: 0.000000; margin mean/min/median/max: -0.231607/-7.827654/-0.268394/6.603231

Deterministic representative sample:

PROMPT:
Ben quickly picked up the green hat. Sam left the green toy on the table.
Who picked up the green hat?

CORRECT CANDIDATE:  Ben.
COMPETING CANDIDATE:  Sam.
CANDIDATE LOG-LIKELIHOODS: [-18.23421896072794, -13.855908425137212]
MARGIN: -4.378310535590728
SELECTED ANSWER:  Sam.
RESULT: incorrect

### surface_form
Items: 64; item accuracy: 0.546875; ties: 0; complete families: 0/8; mean within-family reversal success: 0.093750; margin mean/min/median/max: -0.127963/-6.812221/0.268388/5.952355

Deterministic representative sample:

PROMPT:
The green ball was found by Ben. The green fish was carried by Tim.
The person who found the green ball was

CORRECT CANDIDATE:  Ben.
COMPETING CANDIDATE:  Tim.
CANDIDATE LOG-LIKELIHOODS: [-13.22707886862054, -14.96893663259579]
MARGIN: 1.7418577639752506
SELECTED ANSWER:  Ben.
RESULT: correct

### distractor
Items: 64; item accuracy: 0.484375; ties: 0; complete families: 0/8; mean within-family reversal success: 0.062500; margin mean/min/median/max: -0.113985/-10.281071/-0.545509/7.807681

Deterministic representative sample:

PROMPT:
Ben found the blue hat. Sam carried the blue kite. Sam watched a small bird outside.
Who found the blue hat?

CORRECT CANDIDATE:  Ben.
COMPETING CANDIDATE:  Sam.
CANDIDATE LOG-LIKELIHOODS: [-21.82127598987802, -13.77749499445543]
MARGIN: -8.04378099542259
SELECTED ANSWER:  Sam.
RESULT: incorrect

## Naturalistic outputs

PROMPT:
One afternoon, Lily noticed a red ball near the tree.

BABY:


RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Mia opened the old book and found a surprise.

BABY:
 Mia picked up the red book.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Sam carried a blue kite across the yard.

BABY:
 Sam put down the blue kite.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Tom shared his lunch with Nora.

BABY:
 Tom put down the yellow box.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Alex heard a bird outside the window.

BABY:
 Alex looked at the blue bird.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Zoe looked for her missing hat.

BABY:
 Zoe searched for the yellow hat.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Owen built a small box for the toy.

BABY:
 Owen picked up the blue box.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Nora saw dark clouds above the park.

BABY:
 Nora smiled at the green box.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Lily dropped her green ball beside the bench.

BABY:
 Lily put down the blue ball.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Mia followed the sound of music down the hall.

BABY:
 Mia played with the red hat.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Sam found a tiny fish in the shallow stream.

BABY:
 Sam picked up the yellow fish.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Tom helped Alex fix the broken car.

BABY:
 Tom put down the blue car.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Zoe brought a yellow kite to the beach.

BABY:
 Zoe picked up the yellow kite.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Owen placed the book beside his bed.

BABY:
 Owen had fun with the blue book.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Nora waited quietly for her friend.

BABY:
 Nora put down the yellow ball.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Alex noticed that the room was getting dark.

BABY:
 Alex put down the green box.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Lily heard a knock at the door.

BABY:
 Lily looked at the red kite.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Mia picked up the toy and smiled.

BABY:
 Mia carried the red toy.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Sam walked home after the rain stopped.

BABY:
 Sam carried the red car.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Tom gave Nora a warm blanket.

BABY:
 Tom put down the yellow ball.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Zoe watched the little bird fly away.

BABY:


RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Owen found a new path through the garden.

BABY:
 Owen picked up the green ball.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Nora carried the box carefully upstairs.

BABY:
 Nora put down the blue car.

RUBRIC:
HUMAN REVIEW REQUIRED

PROMPT:
Alex decided to share the red ball.

BABY:
 Alex put down the red ball.

RUBRIC:
HUMAN REVIEW REQUIRED

## Interpretation

These results characterize only the frozen constructions. They do not establish broad English competence, conversation, general reasoning, or a causal effect of P7 training. Failures do not establish architectural impossibility. Exact-string non-overlap does not exclude semantic or near-duplicate contamination.
