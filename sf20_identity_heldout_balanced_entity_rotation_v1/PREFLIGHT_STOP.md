# SF20 pre-seal stop

**Status:** `SF20_PRESEAL_HARD_STOP_TOKENIZER_ELIGIBILITY`

SF20 cannot be built literally with the frozen tokenizer. The treatment requires 16 new defensible person names that are each one tokenizer token at both sentence-start and answer boundaries. Exhaustive inspection of all 1,024 decoded vocabulary tokens found 17 capitalized alphabetic singleton tokens:

`The`, ` The`, ` Ne`, ` That`, ` An`, ` Separating`, ` Nearby`, ` It`, ` They`, ` As`, ` Words`, `So`, `What`, ` One`, ` Once`, ` Before`, ` Then`.

None is an unambiguous ordinary person name available as one token at both required boundaries. A fixed reference list of 50 common names produced **0 eligible names**, versus 16 required.

The existing four names do not satisfy the proposed constraint either:

| Name | Sentence-start IDs | Leading-space IDs | Candidate with period |
|---|---|---|---|
| Alex | 37, 290, 92 | 314, 290, 92 | 314, 290, 92, 18 |
| Owen | 51, 91, 274 | 536, 91, 274 | 536, 91, 274, 18 |
| Mia | 49, 77, 69 | 925, 77, 69 | 925, 77, 69, 18 |
| Nora | 50, 276, 69 | 512, 276, 69 | 512, 276, 69, 18 |

Every tested name round-tripped exactly, so this is not an encoding defect. It is a vocabulary-granularity limitation.

The tokenizer hash matched `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`. All three intended SF8 parent hashes also matched their sealed records. No checkpoint was loaded, no optimizer was created, no model inference or evaluation scoring occurred, and no training update ran.

Proceeding would require a scientific amendment defining multi-token name eligibility, length/boundary matching, and assignment constraints, or changing the tokenizer/identity definition. Those alternatives are materially different and were not chosen silently.

No SF20 treatment dataset, seed mapping, controller, or sealed execution bundle was produced.
