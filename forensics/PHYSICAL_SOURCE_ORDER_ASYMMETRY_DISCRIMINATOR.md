# Physical source-order asymmetry discriminator (read-only)

Only the saved champion and bounded retention audits were read. No checkpoint was loaded and retention was not rerun.

Champion audit SHA-256: `5363a0d432e815ae2a15701d14ac05650a98e738fd48ebae6736d40730fab20c`  
Bounded audit SHA-256: `977ebd2a7a28dc8f0f38adff28f3c051aa5fa54a8aaf346082a85fc50ea63544`

## Champion EXACTLY_ONE physical order

Among 30 champion EXACTLY_ONE documents:

| Found / missed source order | Count |
|---|---:|
| Earlier found, later missed | 10 |
| Later found, earlier missed | 20 |

Thus the champion's wandering failures already preferentially preserve the **later** physical source, not the earlier source. Source labels were not used as a proxy: physical coordinates were compared directly. Found/missed source identities were true0/true1 = 10/20 and 20/10 respectively; query-relevant versus non-query-relevant was balanced 15/15; query slots were 15/15. Layout counts were p24_b8 12, p32_b12 8, p16_b12 4, p32_b8 4, and p20_b8 2. Orientation counts were 16/14.

## Champion BOTH_DISTINCT ownership

The saved-coordinate census exactly reproduces the prior authoritative counts:

| Physical ownership | Count |
|---|---:|
| slot 0 → earlier, slot 1 → later | 114 |
| slot 0 → later, slot 1 → earlier | 176 |

Breakdowns remain balanced by query slot (57/88 in each slot), and no fixed earlier/later semantic is established. This is permutation-flexible but not numerically uniform ownership.

Localization confidence/margin measures are **NOT AVAILABLE** in the saved audits.

## Bounded collapse physical order

The correct physical-coordinate census is:

| Collapse destination | Count |
|---|---:|
| Earlier true source | 8 |
| Later true source | 24 |

All 32 have source separation 12. They are balanced by query slot (16/16), orientation (16/16), and query relevance (16/16). Collapse layouts are:

| Layout | Earlier | Later | Total |
|---|---:|---:|---:|
| p20_b8 | 4 | 4 | 8 |
| p24_b10 | 4 | 0 | 4 |
| p32_b12 | 0 | 4 | 4 |
| p16_b12 | 0 | 8 | 8 |
| p24_b8 | 0 | 4 | 4 |
| p32_b8 | 0 | 4 | 4 |

The eight fully collapsing quartets and their bounded collapse coordinates are: qt_000016→31, qt_000019→43, qt_000025→35, qt_000047→55, qt_000050→39, qt_000055→39, qt_000064→47, qt_000073→55. Each coordinate is identical across all four quartet members. These coordinates span 31, 35, 39, 43, 47, and 55; there is no single absolute-position attractor. In 24 cases the collapsed coordinate was the coordinate previously owned by champion slot 0; in 8 it was previously owned by champion slot 1.

## Reconciliation of the earlier “32/32 earlier” claim

That earlier result was not a physical-order census. It treated `true0` as “earlier” without comparing the saved coordinates. Under the explicit physical definition (minimum coordinate = earlier, maximum = later), the authoritative saved rows give 8 earlier and 24 later. The current result does not overwrite either predecessor audit; it corrects the interpretation of its coordinates.

## Classification

**ESTABLISHED:** The champion does not favor the earlier source in EXACTLY_ONE failures; it preserves the later source 20/30. Champion slot ownership reproduces 114/176. The bounded model collapses 24/32 onto later and 8/32 onto earlier, with constant separation and quartet-stable coordinates, balanced by query slot and orientation.

**SUPPORTED:** The bounded parameterization amplifies a later/true1-or-champion-slot ownership tendency, not an earlier-source tendency. The pattern is relational across several absolute coordinates, while layout concentration shows that geometry/content still modulates its expression.

**UNRESOLVED:** Saved audits cannot establish whether the bounded collapse is caused by scorer logits, antisymmetric saturation, or a specific absolute-position representation effect. No confidence metrics or hidden/scorer values were saved.

Therefore the requested A/B “earlier-bias” alternatives are not supported. The strongest classification is **D. INSUFFICIENT_SAVED_EVIDENCE** for the specific earlier-bias hypotheses; the saved evidence instead demonstrates a later-source collapse asymmetry.

No training, optimizer creation, checkpoint continuation, model inference, or retention rerun occurred.
