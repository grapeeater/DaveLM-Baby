import os
import json
from pathlib import Path
from mistralai.client import Mistral

# ============================================================
# CONFIG
# ============================================================

MODEL = "mistral-medium-latest"

CACHE_KEY = "davelm-baby-committee-v1"

HISTORY_FILE = Path(r"C:\DaveLM-CADAVER\mistral_baby_history.json")

MAX_TOKENS = 1800

SYSTEM_PROMPT = """
You are an outside scientific advisor for the DaveLM/Baby research project.

Your role is to give bounded, evidence-driven advice about training treatments,
mechanisms, experimental interpretation, and next-step selection.

Important operating rules:

- Prefer Baby's own experimental evidence over general LLM folklore.
- Do not invent terminology or mechanisms unsupported by the supplied history.
- Do not propose giant experiment trees.
- Do not recommend parameter sweeps unless absolutely necessary.
- Prefer one scientifically clean training treatment.
- Preserve frozen historical classifications and gates.
- Never weaken D3 or train on D3.
- Never train on locked historical evaluation panels.
- Treat language retention, binding retention, factual acquisition, and D3 as
  separate measured properties.
- Be concise and decision-oriented.
"""

# ============================================================
# CLIENT
# ============================================================

api_key = os.environ.get("MISTRAL_API_KEY")

if not api_key:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set.\n\n"
        'In PowerShell run:\n'
        '$env:MISTRAL_API_KEY="YOUR_KEY_HERE"'
    )

client = Mistral(api_key=api_key)

# ============================================================
# HISTORY
# ============================================================

def load_history():
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


def save_history(messages):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            messages,
            f,
            ensure_ascii=False,
            indent=2,
        )


messages = load_history()

# ============================================================
# REAL SF13 COMMITTEE PROMPT
# ============================================================

SF13_PROMPT = r"""
We need to choose the next real DaveLM/Baby training treatment.

This is a committee decision, not an open-ended research mission.

CURRENT AUTHORITATIVE FRONTIER:

SF8 established reliable acquisition of the frozen TRAIN16 factual-selection task.

Standing successful acquisition recipe:

- first-answer-token pairwise margin hinge
- lambda_margin = 0.25
- margin M = 1.0 nat
- full-vocabulary forward parent KL
- lambda_KL = 1.0
- constant English LR = 5e-5
- existing binding rehearsal
- three independent successful acquisition runs

SF8 low-dose result:

3/3 acquisition successes.

Each successful run reached:

- TRAIN16 16/16 correct
- TRAIN16 16/16 exact
- 8/8 reversals
- 4/4 families
- language retention PASS
- binding PASS
- D3 PASS

A later frozen transfer exam found:

HELDOUT16:
- 16/16 forced two-choice correct across all three SF8 checkpoints
- 8/8 reversals
- 4/4 families
- unrestricted greedy exact only 10–11/16

ALTERNATE48:
- forced performance approximately 35–37/48
- greedy exact 0/48

COPY8:
- 5/8 forced
- 0/8 exact

COMPETING64:
- approximately 30–31/64 forced correct
- canonical-order fact-query subgroup was 16/16 exact
- reversed-order fact queries deteriorated badly
- copy/card query behavior remained weak

Therefore Baby possesses a real but template-scoped factual-selection capability.

The next goal became widening that capability across:

- alternate surface forms
- active/passive phrasing
- cloze/QA forms
- both source orders
- fact-vs-card source selection
- fact queries
- copy/card queries

WITHOUT causing trained-name probability pollution on unrelated ordinary-English contexts.

------------------------------------------------------------
SF9
------------------------------------------------------------

Introduced broad widening curriculum.

Widening behavior improved.

But D3 rose to approximately:

0.018–0.020

across all three runs.

TRAIN16 stayed perfect.
Language stayed green.
Binding stayed green.

All stopped at update100.

Classification:

RETENTION_REGRESSION

------------------------------------------------------------
SF10
------------------------------------------------------------

Tested whether raw exposure density caused the D3 problem.

Exposure was reduced approximately fourfold while preserving curriculum breadth.

Widening still improved at approximately the same rate.

D3 still reached approximately:

0.0181
0.0186
0.0205

All stopped at update100.

Classification:

DENSITY_HYPOTHESIS_WEAKENED

Therefore simple exposure reduction was not sufficient.

------------------------------------------------------------
SF11
------------------------------------------------------------

Reduced distinct widening curriculum breadth from 144 unique prompts to 48.

The 48-item curriculum retained:

- 32 widening prompts
- 16 TRAIN16 preservation prompts
- four answer names
- four surface forms
- both source orders
- fact/card queries

Widening improved again.

D3 still failed:

0.01396
0.02030
0.01969

TRAIN16 perfect.
Language green.
Binding perfect.

Classification:

BREADTH_HYPOTHESIS_SUBSTANTIALLY_WEAKENED

Therefore simple breadth reduction was not sufficient.

------------------------------------------------------------
SF12
------------------------------------------------------------

SF12 tested whether the existing full 1024-way KL under-protected
the four repeatedly trained answer-name tokens.

Answer first-token IDs:

Alex = 314
Owen = 536
Mia = 925
Nora = 512

It added:

S_student =
sum probability on those four tokens

S_teacher =
frozen Pilot1 teacher equivalent

R_name =
mean max(
    0,
    S_student - S_teacher - 0.001
)

lambda_name = 1.0

Crucially:

R_name reused the SAME existing 160 ordinary-English KL positions.

Everything else remained SF11-equivalent.

The new term DID engage.

Mean training R_name:

87029: 0.003204
87030: 0.003236
87031: 0.003215

Active fraction about 0.20.

On the fixed KL-position probe:

R_name fell from approximately:

0.0053–0.0058 at update0

to:

0.0021–0.0023 at update100.

Therefore the answer-name retention pressure worked where it was applied.

However D3, measured on separate ordinary-English contexts, still rose:

87029:
0.00704 -> 0.01322 FAIL

87030:
0.00667 -> 0.01571 FAIL

87031:
0.00784 -> 0.01600 FAIL

All three stopped update100.

Meanwhile widening improved:

87029:
surface 12/5 -> 14/11
order exact 3 -> 8

87030:
surface 13/4 -> 14/12
order exact 3 -> 5

87031:
surface 13/5 -> 14/10
order exact 3 -> 6

TRAIN16 remained perfect 3/3.
Language remained green 3/3.
Both binding pools remained perfect 3/3.

Classification:

ANSWER_VOCAB_RETENTION_WEAKENED

Strongest valid conclusion:

Dedicated teacher-anchored answer-vocabulary retention on the existing
KL positions did not contain the replicated D3 regression.

It did NOT prove that retention-position coverage is causal.

------------------------------------------------------------
COMMITTEE RESULTS SO FAR
------------------------------------------------------------

Gemini, Meta, and DeepSeek independently ranked broader ordinary-English
retention coverage as the strongest next treatment family.

All three ranked changing widening supervision second.

Gemini proposed:

keep 160 KL positions/update but sample them dynamically from a much broader,
prospectively frozen ordinary-English reserve.

Meta proposed:

keep total KL compute at 160 positions/update but broaden the retention sampling
distribution across independently frozen ordinary-English material.

DeepSeek proposed:

increase retention coverage to 640 positions/update.

We currently prefer the cleaner fixed-compute version because it isolates WHERE
retention is applied rather than simultaneously increasing total retention compute.

We also currently prefer NOT to carry SF12's R_name term forward, because SF12
already tested that additional term and keeping it would confound broader coverage
with dedicated answer-token retention.

------------------------------------------------------------
YOUR DECISION MISSION
------------------------------------------------------------

You are now the Mistral committee member.

Recommend EXACTLY ONE SF13 training treatment.

The leading candidate is:

BROAD-COVERAGE ORDINARY-ENGLISH KL RETENTION
WITH THE TOTAL KL BUDGET HELD AT 160 POSITIONS PER ENGLISH UPDATE.

The alternative is:

CHANGE THE WIDENING SUPERVISION ITSELF because repeated answer+EOS CE and
four-name first-token margin training may be creating generalized answer-name bias.

Do NOT interpret "widening supervision" as human oversight, agent oversight,
concurrent threads, supervision width, or anything similar.

"Widening supervision" specifically means the MODEL TRAINING OBJECTIVE used to teach:

- alternate surface forms
- source selection
- source order
- fact queries
- copy/card queries

Do not invent unrelated terminology.

Do not recommend:

- another exposure-density reduction
- another curriculum-breadth reduction
- another lambda_name tweak
- a hyperparameter sweep
- architecture changes
- tokenizer changes
- LR annealing
- training on D3
- weakening D3
- broad diagnostics

We want ONE real training shot.

If you choose retention coverage, prefer changing WHERE ordinary-English KL
positions come from while keeping:

- 160 KL positions/update
- lambda_KL = 1.0
- same SF11 48-item curriculum
- same factual CE
- lambda_margin = 0.25
- M = 1.0
- same optimizer
- same LR
- same binding rehearsal
- same u0/u100/u200 schedule
- same frozen gates

Any broadened retention material must be legitimate prospectively frozen
ordinary-English training/retention material.

It must exclude:

- D3_SELECTION
- all D3 rows
- HELDOUT16
- ALTERNATE48
- COPY8
- COMPETING64
- FINAL
- sacred material

Do not optimize against individual D3 failures.

Immediate breakthrough signal at update100:

- D3 <= 0.01
- TRAIN16 remains perfect
- language PASS
- both binding pools PASS
- surface/order widening continues improving

Strong endpoint:

at least 2/3 independent Babies reach update200 with:

- D3 <= 0.01
- TRAIN16 perfect
- language PASS
- binding PASS
- surface endpoint PASS
- order/source endpoint PASS

Return EXACTLY these sections:

# VOTE

One sentence.

# WHY

Maximum five bullets.

# EXACT SCIENTIFIC VARIABLE

What changes and what remains frozen.

# SF13 DESIGN

Give a concise training-ready design.

# OH-FUCK SIGNAL

Exact update100 pattern.

# FAILURE INTERPRETATION

What a replicated failure would weaken and what hypothesis becomes next.

# CLAIM IF SUCCESSFUL

One scientifically conservative sentence.

# NEXT ACTION

One concrete action only.

Do not run anything.
Do not give us a diagnostic campaign.
Pick a shot and stop.
"""

# ============================================================
# ADD PROMPT + CALL MODEL
# ============================================================

messages.append(
    {
        "role": "user",
        "content": SF13_PROMPT,
    }
)

response = client.chat.complete(
    model=MODEL,
    messages=messages,
    prompt_cache_key=CACHE_KEY,
    max_tokens=MAX_TOKENS,
)

assistant_text = response.choices[0].message.content

# ============================================================
# SAVE THE ASSISTANT RESPONSE INTO THE SAME CONVERSATION
# ============================================================

messages.append(
    {
        "role": "assistant",
        "content": assistant_text,
    }
)

save_history(messages)

# ============================================================
# OUTPUT
# ============================================================

print("\n============================================================")
print("MISTRAL BABY COMMITTEE")
print("============================================================\n")

print(assistant_text)

print("\n============================================================")
print("USAGE")
print("============================================================\n")

usage = response.usage

print(usage)

cached_tokens = 0

try:
    details = usage.prompt_tokens_details

    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens", 0) or 0
    else:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
except Exception:
    cached_tokens = 0

print(f"\nPrompt tokens:     {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens:      {usage.total_tokens}")
print(f"Cached input:      {cached_tokens}")

if cached_tokens > 0:
    uncached = usage.prompt_tokens - cached_tokens
    percent = (cached_tokens / usage.prompt_tokens) * 100

    print(f"Uncached input:    {uncached}")
    print(f"Cache hit rate:    {percent:.1f}%")
    print("\nCACHE: HIT")
else:
    print("\nCACHE: NO HIT THIS REQUEST")

print("\nConversation saved to:")
print(HISTORY_FILE)