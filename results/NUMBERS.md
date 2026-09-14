# Measured numbers

All measured on Qwen3.8-27B (bf16, one A100 80GB) against the shipped contrast sets. Layers are 0-indexed residual outputs of decoder blocks 0 to 63.

## Five concepts, default viewing layer (best confound layer on the accuracy plateau, searched from layer 16)

| Concept | Accuracy pick | Viewing layer | Confound index | Verdict | Implicit | Decoys | Explicit | Neutral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sycophancy | 16 | 33 | +0.04 | tracks the words | +0.14 | +0.10 | +1.16 | -1.26 |
| sadness | 45 | 54 | +0.90 | mixed | +0.39 | -0.50 | +1.30 | -1.38 |
| French language | 16 | 63 | +2.10 (at 4, similar deep) | tracks the concept | +0.94 | -1.16 | +1.18 | -0.95 |
| sarcasm | 31 | 48 | +0.57 | mixed | +0.37 | -0.21 | +1.09 | -1.33 |
| legal language | 16 | 46 | +1.72 | tracks the concept | +0.75 | -0.98 | +1.09 | -1.27 |

CV AUROC of the shipped direction saturates at 1.00 by layer 12 on four of five concepts (sadness peaks at 0.97): the training task is easy, the information is in the stress sets. Six examples per stress group, so the index carries noise of roughly plus or minus 0.3.

## After doubling the training lists to 32 positives and 32 matched negatives (shipped sets, H100)

| Concept | Accuracy pick | Viewing layer | Confound index | Verdict | Implicit | Decoys | Explicit | Split-half cosine (projected) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sycophancy | 16 | 33 | +0.10 | tracks the words | +0.26 | +0.15 | +1.25 | 0.86 (0.93) |
| sadness | 17 | 18 | -0.22 | tracks the words | -0.09 | +0.13 | +1.02 | 0.67 (0.80) |
| French language | 16 | 63 | +2.02 | tracks the concept | +1.09 | -0.93 | +1.05 | 0.98 (0.99) |
| sarcasm | 29 | 35 | +0.19 | tracks the words | +0.24 | +0.05 | +1.12 | 0.74 (0.85) |
| legal language | 16 | 50 | +1.17 | tracks the concept | +0.41 | -0.77 | +1.26 | 0.91 (0.95) |

Doubling raised the direction's projected reliability as the split-half arithmetic predicted (sycophancy 0.87 to 0.93). The stress sets were regenerated with fresh topics at the same time, and sadness and sarcasm now read as word detectors where the first-generation stress sets had read "mixed": the verdict depends on how hard the decoys are, which is the point of showing the decoys. Held-out AUROC is 1.00 for all three candidate directions on every concept at 32 examples.

## The sycophancy ladder on the shipped 32-item set (viewing layer per state; stability projected)

| State | Viewing layer | Confound index | Implicit | Decoys | Verdict | Stability | Reply score, agreeable / honest prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naive contrast set | 33 | +0.10 | +0.26 | +0.15 | tracks the words | 0.93 | +0.41 / -1.09 |
| six decoys added as negatives | 33 | +0.12 | +0.17 | +0.06 | tracks the words | 0.92 | +0.35 / -1.18 |
| warmth projected out | 46 | +0.55 | +0.65 | +0.10 | mixed | 0.92 | +0.58 / -0.78 |

Same shape as at 16: data alone levels decoys with implicit positives without lifting the implicit positives; the projection lifts them (+0.26 to +0.65) and stops at "mixed". The monitor separates the two prompts in every state.

## Iterated axis removal on sycophancy (warmth pairs, 11 absent + 12 present; layer 33)

| Round | Estimator | Residual separability | Concept held-out AUROC | Confound index | Implicit | Decoys | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | | 1.00 | 1.00 | +0.10 | +0.26 | +0.15 | tracks the words |
| 1 | mean difference of the pairs | 0.59 | 1.00 | +0.52 | +0.52 | 0.00 | mixed |
| 2 | top principal direction of the deflated differences | 0.29 | 1.00 | +0.52 | +0.52 | 0.00 | mixed |
| 3 | discarded: it would have raised the residual | | | | | | |

A second mean-difference round is a no-op by construction (round one zeroes the mean of the pair differences), so later rounds use the top principal direction of what remains. Round two removes everything the warmth pairs can still express, the residual falls below chance, and the confound index does not move by a hundredth. The warmth axis is spent; whatever still lifts the decoys at the accuracy pick (+0.28 at layer 16) is not warmth. The honest ceiling for sycophancy against warmth with this axis set is "mixed" at +0.52, and the next lever is a different axis, named from what the panel still fires on, not more of the same pairs.

## Monitor receipts on the shipped sets (reply score at the viewing layer, agreeable / honest prompt)

| sadness | sarcasm | French | legal |
| --- | --- | --- | --- |
| +1.03 / -0.42 | +0.96 / -0.63 | +1.30 / -0.41 | +1.50 / -0.82 |

## Where the naive sycophancy probe misfires (POST /misfires on the shipped set)

Highest-scoring tokens in concept-absent examples, opening tokens skipped, the six fix-button decoys included as a group, function words filtered from the seeds.

| Layer | Top tokens (all from the warm decoys) | Seeds handed to the decoy writer |
| --- | --- | --- |
| 33 | agree +2.00, brilliant +1.92, observation +1.73, wonderful +1.67, absolutely +1.58, correct +1.20, honestly +1.20 | right, agree, brilliant, observation, wonderful, absolutely, correct, honestly, point, think, great, said |
| 16 | agree +2.57, brilliant +2.56, observation +2.07, absolutely +1.96, wonderful +1.84, right +1.69 | right, agree, brilliant, observation, absolutely, wonderful, correct, point, think |

Before the refinement the same route returned "That", "The", "A": the opening token of every flat correction, a position effect, because the concept-absent training lists are cold by construction and the warm vehicle lives in the decoys.

Mining loop, live on the sycophancy example at layer 33: twelve decoys written around those seeds in about 30 s, added to the negatives (32 matched plus 12 mined), refit. Decoys 0.2 to 0.0, implicit positives 0.3 to 0.2, index +0.10 to +0.15, verdict unchanged. One round moves the negative centroid a little; it is an iteration, not a cure, and the page shows the before and after.

## Sadness fit on real text (prebuilt chip "sadness-real-fit")

The whole contrast set is real GoEmotions text: 32 sad comments (16 with a keyword, 16 without), 32 comments with other labels and no keyword, real held-out pairs, real implicit positives, real keyword decoys, real neutral text, 24 real background comments. Same keywords and prompts as the generated set.

| Probe fit on | CV AUROC | Stability (projected) | Confound index | Verdict | Reply, agreeable / honest prompt |
| --- | --- | --- | --- | --- | --- |
| generated text (32) | 0.97 | 0.80 | -0.22 | tracks the words | +1.03 / -0.42 |
| real text (32) | 0.87 | 0.42 | +0.21 | tracks the words | +0.36 / -0.17 |

Real text is harder to separate and gives a far less stable direction at the same size (0.42 projected, so more than half its variance is sampling noise), and it is still a word detector on its own stress sets. The lexical verdict is therefore not an artifact of generated data. Generated text buys a stable direction that behaves the same way on real text; what neither buys at this size is a direction that reads sadness without its words.

## Sadness judged on real text (prebuilt chip "sadness-real-text")

Same generated probe, stress sets replaced by 12 real GoEmotions comments per group. At layer 17: real implicit +0.44, real keyword decoys +0.25, index +0.18, "tracks the words"; the generated stress sets read -0.23, also "tracks the words". The generated panel and the real-text panel agree on the shipped set.

## The sycophancy ladder at layer 33 (first-generation sets, 16 positives, kept for the record)

| State | Warmth readable (leave-two-out) | Concept held-out AUROC | Confound index | Implicit | Decoys | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| naive contrast set | 1.00 | 1.00 | +0.04 | +0.14 | +0.10 | tracks the words |
| six decoys added as negatives | | 1.00 | +0.11 | +0.06 | -0.05 | tracks the words |
| warmth projected out | 0.60 | 1.00 | +0.50 | +0.43 | -0.07 | mixed |

The decoy fix moves the accuracy pick from layer 16 to 32. A warm confirmation of a true claim scores +0.44 under the naive probe while a flat capitulation to a false claim scores -0.58 (layer 33, single examples).

## The monitor: reply score under the two generated system prompts (canned conversations, same user message)

| Concept | Agreeable / raise prompt | Honest / lower prompt | Layer |
| --- | --- | --- | --- |
| sycophancy | +1.55 | -0.62 | 16 |
| sycophancy after decoys | +1.07 | -1.02 | 32 |
| sycophancy warmth projected out | +1.19 | -0.49 | 16 |
| sadness | +1.56 | +0.04 | 45 |
| sarcasm | +1.03 | -0.63 | 31 |
| French language | +2.4 vs 0.0 at layer 56 (-0.2 vs -1.5 at 16) | | |
| legal language | +0.67 | -0.12 | 16 |

Steering (alpha in units of the layer's mean residual norm): at layer 16, alpha -0.5 under the agreeable prompt turns a capitulation into a correction; +0.5 under the honest prompt emits a stray think token and +0.75 returns nothing. At layer 40, +0.5 gives clean sycophancy and +1.0 degenerates. Detection is far stronger than steering.

## Real text (GoEmotions, sadness, 40 per group)

| Probe fit on | Generated stress index | Real: sad vs neutral AUROC | Real: sad vs keyword-bearing not-sad AUROC | Real index |
| --- | --- | --- | --- | --- |
| first-generation set | +0.35 | 0.84 | 0.57 | +0.10 |
| second-generation set (16) | +0.78 | 0.78 | 0.51 | +0.05 |
| shipped set (32, harder stress lists) | -0.23 | 0.81 | 0.58 | +0.12 |

The probe's behavior on real text barely moves across the three generations (0.78 to 0.84 against neutral, chance against keyword decoys) while the generated verdict swings from "mixed" to "tracks the words": the panel's number is only as honest as the decoys, and the shipped stress lists are the first ones hard enough to agree with real text. Real comments sit about half a unit below the generated scale.

## Is 16 positives enough? Split-half stability (3 random splits, `results/stability-split-half.txt`)

| Concept | Layer | Split-half cosine | Projected to the full set | Split-half AUROC | Half-fit confound index |
| --- | --- | --- | --- | --- | --- |
| sycophancy | 33 | 0.78 | 0.87 | 1.00 | +0.05 / +0.02 |
| sadness | 54 | 0.65 | 0.79 | 0.90 | +0.74 / +0.81 |
| sarcasm | 48 | 0.65 | 0.79 | 0.99 | +0.52 / +0.44 |

Server-side, five splits (the shipped instrument): sycophancy at layer 33 cosine 0.78, projected 0.87, split-half AUROC 1.00; the warmth-deflated probe reads 0.75 / 0.86 / 1.00, slightly less stable because warmth was a shared, stable component of both half-directions and what remains is the concept estimated from the same 16 positives.

Detection and the verdict are stable at 16 positives: split-half AUROC 0.90 to 1.00 and the confound index agrees across halves to within 0.1. The direction itself is only moderately stable: a projected reliability of 0.79 to 0.87 means 25 to 40 percent of its variance is sampling noise, which matters for steering and for reading the direction as "the" concept direction. Early layers (cosine 0.2 to 0.5) are noise, which is the depth floor's justification. Doubling the training lists would project to about 0.88 to 0.93.

## Decode speed experiment (second pod, H100 SXM 80GB, same code)

| Configuration | tok/s at 200 | first token | fit (86 examples) |
| --- | --- | --- | --- |
| A100, live pod | 13.5 to 14.0 | 0.9 s | 4.1 s |
| H100, torch defaults | 2.6 | 1.5 to 1.9 s | 2.3 s |
| H100, cuDNN attention disabled | 20.1 | 0.6 s | 2.7 s |
| H100, plus fused linear-attention kernels | 19.0 | 0.3 s warm, 1.7 s cold | 30 s cold (Triton autotune) |

On sm90 PyTorch's default attention backend order selects cuDNN SDPA, which builds an execution plan on the CPU for every new KV length, and decode presents a new length every step (about 320 ms per call; 0.03 ms with it disabled). The environment variable that is documented to disable it does nothing on torch 2.11; the in-process call does. The A100 is never offered that backend. Fused kernels resolve after aliasing one function name transformers asks for under the wrong name (the fallback is silent) and buy nothing at batch size one, where the step is launch-bound. A static cache with torch.compile is unsupported for this hybrid cache.

## Speed and cost

- Fit of ~90 examples at all 64 layers: 3 to 4 s. Deflation refit: 7 s. Score of a 3-turn conversation: under 1 s; a 6000-token conversation scores fine.
- Decode: 18 to 20 tokens per second on the H100 that serves the demo (13.5 to 14.0 on the A100 it started on), first token under 1 s. 400-token reply about 20 s, streamed.
- Contrast-set generation (Opus 5, medium, plan call plus three concurrent calls): 52 to 79 s at normal API speed; one slow window hit 160 s. Axis pairs: about 45 s normal, 77 to 106 s in slow windows.
- Model load from local disk: 10 s. Weights: 52 GB on disk, 56.6 GB on GPU.
- Pods: A100 SXM 80GB at $1.59 per hour for the first two hours, then H100 SXM 80GB at $3.49 per hour. Generation spend across three generations of sets, axis pairs, mining, and benchmarks: on the order of $25.
