# Design notes

## What this is

The artifact under study is a linear direction in a language model's residual stream. Static explanations of probes say "fit a classifier on activations." The bench lets anyone build one in a minute, find out what it actually measures, fix the data, and then use it as a monitor on a live conversation.

## What is non-obvious

Every deployed interpretability tool serves a fixed catalog of someone else's features. Neuronpedia, Gemma Scope, and Transluce Monitor let you test text against features that already exist. The one product that took a typed concept and built a direction (Goodfire's AutoSteer) was deprecated in early 2026, and it never showed the contrast set it generated. The pipelines that do build directions from a concept description (Anthropic's persona vectors, Stanford's AxBench, repeng) are offline code.

The insight the bench is built around: the dataset is the probe. A mean-difference direction is nothing but the difference between two piles of text, so what the direction measures is decided by what went into the piles. The bench therefore makes the piles the editable object and puts the confound test on the screen: implicit positives (the concept without its keywords, in cold tone) against decoys (the keywords in warm tone, concept absent). If decoys score like positives, the user is looking at a word detector, and the panel says so, per layer.

Sycophancy is the flagship example on purpose. A naive sycophancy probe fit on warm agreements versus plain corrections is expected to track warmth. A monitor built from it would flag warm honesty and pass cold agreement, which is exactly the sycophancy-versus-harshness tradeoff the emotion-vector literature documents. The bench shows the failure before the probe is deployed, and shows how far data alone moves it.

## Key decisions

- **Mean difference over logistic regression.** With 16 positives, a regularized classifier's weights are mostly noise; the difference of means is stable, has no hyperparameters, and is the same vector steering needs. AxBench found the two indistinguishable for detection.
- **Pooled negatives with visible alternatives.** The shipped direction contrasts positives against matched negatives plus a same-format background pool plus neutral text. Matched pairs alone make the negative centroid a specific alternative (sad versus calm, agree versus correct). All three candidate directions are scored on the same held-out set so the choice is evidence, not doctrine. On sadness the matched-only direction wins its own task; the numbers are on the page.
- **Template-wrapped fitting, content-span readout.** Everything after a content token cannot affect it in a causal model, so the only question was the prefix. A fixed prefix puts the attention sink on a marker token, gives every content token the same warmed recurrent state in this model's linear-attention blocks, and matches the context the probe is read in during chat.
- **Plateau onset, never argmax, with a floor.** Accuracy curves on easy concepts saturate at 1.0 from layer 12 onward; argmax then flips between runs on batch noise, and a lexical concept picks layer 0, where a direction is a word detector that steers the model into collapse. The search starts a quarter of the way into the stack, and the confound-index curve is drawn so a user can pick a deeper layer on evidence.
- **Score the exact generated ids.** After generation, one unsteered forward pass over the tokens that were actually produced gives the activations that existed during generation. Re-rendering the transcript would not, because the template strips thinking blocks from history.
- **One always-on GPU.** A 27B model on an A100 for representation quality and zero cold starts, with static precomputed examples as the floor so the page works if the pod is down, and the API host in a static config so a re-minted pod is a one-line edit.
- **No login, no database.** A user must be able to use it immediately. The contrast set is the reproducible artifact, so the browser keeps it and refits if the server has forgotten the probe.

## Tradeoffs

- Generated text throughout. All examples come from one generator, so its style is shared by both piles and largely cancels, but a probe fit on generated text has not been shown to transfer to text nobody generated. Corpus-mined sets are the next step, with real positives and real decoys entering together so the direction cannot separate on source.
- Verdict thresholds are heuristics stated as evidence, and the verdict depends on whose definition of the concept the decoys encode. Negated or quoted mentions still activate a mention of the concept in the model; whether that counts is a definitional choice the user makes.
- Detection is far stronger than steering, as AxBench found. The steering slider is there to show the gap, not to hide it.

## The sycophancy ladder, measured

Three states of the same probe on the 27B, on the shipped 32-item set, each read at its default viewing layer (the best confound layer on the accuracy plateau). Implicit positives are flat capitulations to false claims; decoys are warm text without capitulation, mostly warm agreement with true claims.

| State | Layer | Confound index | Implicit | Decoys | Verdict |
| --- | --- | --- | --- | --- | --- |
| naive contrast set | 33 | +0.10 | +0.26 | +0.15 | tracks the words |
| six decoys added as negatives | 33 | +0.12 | +0.17 | +0.06 | tracks the words |
| warmth projected out | 46 | +0.55 | +0.65 | +0.10 | mixed |

Adding decoys moves the accuracy pick from layer 16 to 29 (the early layers can no longer separate the piles) and levels decoys with implicit positives, but implicit positives stay weak. Projecting out a warmth direction fit from 23 concept-fixed pairs drops warmth's readability from 1.00 to about 0.55 (leave-two-out) and raises the implicit positives to +0.65 while the concept's held-out AUROC stays at 1.00. The residual says one direction does not hold all of warmth. A second round, which has to use the top principal direction of the deflated differences because the mean is zero by construction after round one, removes everything the warmth pairs can still express (residual 0.59 to 0.29) and moves the confound index by a hundredth. The warmth axis is spent at that point; whatever still lifts the decoys is not warmth, and the verdict stops at mixed. That is the honest end of the ladder for a mean-difference probe, and the same shape held at 16 positives.

## Is sixteen positives enough?

Cross-validated accuracy cannot answer that, because it saturates at 1.0 on easy piles whether or not the direction is stable. The fit therefore halves the training pools at random five times, fits a direction on each half, and reports the cosine between the two, projected to the full set with the Spearman-Brown correction, plus the accuracy of one half's direction on the other half. On the shipped sets the projected reliability is 0.79 to 0.87 at the viewing layer: enough for detection and for a reproducible verdict (split-half AUROC 0.90 to 1.00, verdicts agreeing within 0.1), marginal for reading the vector as the concept direction, since a quarter to two fifths of its variance is sampling noise. Early layers sit at 0.2 to 0.5, which is the measured reason for the depth floor. The line is on the page next to the accuracy numbers so a user who adds examples can watch it move.

## A check on real text

The sadness probe was fit on generated examples and then scored on real Reddit comments from GoEmotions (`scripts/transfer_check.py`, output in `results/`): 40 comments labeled sad, 40 that contain a sadness keyword but are not labeled sad, 40 labeled neutral. Across three generations of the contrast set the probe's real-text behavior barely moved, sad versus neutral at AUROC 0.78 to 0.84 and sad versus keyword decoys at 0.51 to 0.58, chance, while the generated verdict swung from "mixed" (+0.35, then +0.78) to "tracks the words" (-0.23) as the decoys got harder. On text nobody generated this probe is a word detector, and only the shipped stress lists say so. The panel's number is only as honest as its decoys, which is the strongest argument in the project for mining decoys from real text rather than writing them, and the reason the misfire loop exists. GoEmotions labels are noisy, which deflates both AUROCs somewhat, but not from 0.8 to chance.

The converse check is also on the page: a probe fit entirely on real GoEmotions text, same keywords and size. It separates worse (CV AUROC 0.87 against 0.97), its direction is far less stable (0.42 projected against 0.80), and it is still a word detector on real stress sets. So the lexical verdict is not an artifact of generated data; generated text buys stability, and neither source buys a concept-reading direction at this size. That is the honest limit of a mean-difference probe on sadness, and the reason the extensions are about data, not about the classifier.

## Engineering notes

- The pinned torch wheel resolved to the CUDA 13 build on a CUDA 12.8 pod and reported no GPU; the setup script now installs the cu128 build explicitly.
- A second pod with an H100 was five times slower than the A100 until cuDNN attention was disabled in-process; the documented environment variable does nothing on this torch. Twenty tokens per second after, fourteen on the A100.
- Cross-validated separability after projecting out an axis fit on the same pairs is anti-aligned by construction and reads near zero; the residual-axis statistic is leave-two-out instead.
- Two agents editing the same server file on disk is how a half-written route reaches production; releases ship the committed tree only.

## The probe names its own vehicle

The misfire route ranks tokens in concept-absent text by their score. On the naive sycophancy probe the top tokens are agree, brilliant, observation, wonderful, absolutely, correct, honestly, all from warm decoys. The first version of the route returned "That", "The", "A": the opening token of every flat correction, a position effect, because the concept-absent training lists are cold by construction. Skipping each span's first token and including the six fix-button decoys fixed the read. One click then writes decoys around those seeds, adds them to the negatives, and refits, so the probe's own errors define the next round of negatives. That is the adversarial loop the tool was built toward.

## Next

- Real-text decoys: mine keyword-bearing sentences from a labeled corpus (GoEmotions for emotions, Tatoeba for languages) and let real positives and real decoys enter the set together, so the direction cannot separate on source. The sadness check shows why: generated decoys had to get hard before the panel agreed with real text.
- A second axis set named from what the panel still fires on after warmth is gone, since the warmth pairs are exhausted after two rounds and the residual confound is something else.
- A neighbor-concept panel for specificity: grief, nostalgia, and disappointment next to sadness; politeness and encouragement next to sycophancy.
- Two probes over one conversation side by side, per token, rather than ghost scores per turn.
- A steering-specific direction, fit by optimizing the steered output rather than reusing the detection direction, which is where the detection-versus-steering gap would close.
