# Probe Bench

**The dataset is the probe.**

Type a concept. The bench generates a contrast set, fits a linear direction in a 27B language model, and tells you whether that direction tracks the concept or only the words and tone that usually travel with it. Then chat with the model under a system prompt you can edit, and watch the concept light up token by token in the model's own words. Push the direction during generation and watch it move without the prompt saying anything.

Live demo: https://probe-bench-six.vercel.app (API host in `frontend/public/config.json`). The demo is password protected; ask for the password. Prebuilt examples work even when the model host is down.

## What you can do

1. **Build a probe.** Type a concept such as `sarcasm`, `legal language`, or `sycophancy`. A contrast set is generated: positives, matched negatives, held-out pairs, three stress lists, and a background pool. One forward pass fits a mean-difference direction at every layer.
2. **Read the verdict.** The confound panel compares implicit positives (concept without its keywords, in cold tone) against decoys (keywords in warm tone, concept absent). If decoys score as high as implicit positives, the direction is a word detector, and the panel says so.
3. **Fix the set and refit.** Delete examples, or add the decoys as negatives with one click. Refitting takes a few seconds.
4. **Check the direction is real.** Every fit reports split-half stability: the training pools are halved at random five times, and the cosine between the two half-directions, projected to the full set, says whether more examples would change the direction. Add your own examples to either pile and refit to watch it move.
5. **Scrub the layer.** Every response carries all-layer values, so the slider re-renders heatmaps and verdicts without a network call. The confound index changes sign with depth on several concepts.
6. **Chat and monitor.** Edit the system prompt, send a message, and every turn is colored by the probe. Per-turn scores and a conversation sparkline show the concept over time. Edit any turn in place and the colors recompute; delete the last exchange to branch the conversation from an earlier point.
7. **Remove the confound axis.** Name the tone the stress sets decouple (warmth for sycophancy), and the bench writes pairs that hold the concept fixed and flip only that tone, fits the axis direction, projects it out of every activation, and refits. The panel shows the axis's readability before and after, the concept's held-out accuracy before and after, and the verdict at both. On sycophancy this moves the verdict from "tracks the words" to "mixed" and stops there, honestly. Further rounds remove the top principal direction of what the pairs still express, with a stop rule when the residual stops dropping; on sycophancy a second round drives the residual below chance without moving the verdict, which says the remaining confound is not warmth.
8. **Ask the probe where it misfires.** The highest-scoring tokens in text where the concept is absent are the vehicle a naive probe rides. On sycophancy they are agree, brilliant, wonderful, absolutely. One click writes decoys around them, adds them to the negatives, and refits.
9. **Save and compare.** Save any fitted probe under a name ("humor", "humor, enthusiasm projected out") and switch between saved and prebuilt probes with the same conversation on screen; each turn shows the previous probe's score as a ghost, or pick a second probe to see both colorings stacked per token with both values in the tooltip. Permalinks reload any probe by id.
10. **Steer.** A slider adds the direction to the residual stream during generation. Expect strong detection and weak steering; the gap is part of the lesson.

## How it works

- **Model.** Qwen3.8-27B in bf16 on one H100 80GB, loaded once by a FastAPI server (an A100 works at 14 tokens per second instead of 20). Residuals are read from every decoder block with `output_hidden_states`.
- **Rendering.** Each example is wrapped as a user turn (or a user turn plus assistant reply for response-property concepts such as sycophancy) with thinking disabled, and only the example's own token span is pooled. A fixed prefix keeps the attention sink on a marker token and gives every content token the same warmed recurrent state in the linear-attention blocks. Position 0 is excluded everywhere.
- **Fitting.** Direction = mean(positives) minus mean(negatives), where negatives are the matched negatives plus the background pool plus neutral text. Matched pairs alone make the negative centroid a specific alternative (sad versus calm, agree versus correct); a broad same-format pool moves it toward text in general. The three candidate directions are all scored on the same matched held-out set so the choice is visible.
- **Layer choice.** Plateau onset of the cross-validated AUROC curve, searched from a quarter of the way into the stack, never argmax: argmax flips between runs on a flat plateau, and a lexical concept can saturate the first few layers.
- **Confound index.** Mean sequence score of implicit positives minus mean sequence score of held-out decoys, in units of the training spread. The verdict is evidence, not a ruling, and it is stated per layer.
- **Chat scoring.** The reply is generated, then one unsteered forward pass over the exact token ids that were generated projects every token onto the direction at every layer. The model is causal, so those are the activations that existed during generation.
- **Steering.** A forward hook on the chosen decoder block adds `alpha * mean residual norm * direction` at every position, prefill and decode.
- **Deflation.** The axis direction is the mean over pairs of pooled(high) minus pooled(low); half the pairs have the concept absent and half present, so the concept cancels. Every pooled and per-token activation is projected onto the axis's orthogonal complement before the standard fit and before scoring, so the deflated probe reads consistently in the monitor. Residual axis readability is a leave-two-out statistic, because a cross-validated mean difference on the very pairs that fit the axis is anti-aligned by construction.

## Repository

```
backend/    FastAPI server (server.py), model runner (model.py), probe math (probe.py), contrast-set generator (generate.py), wire contract (contract.py), config (config.py), smoke test (smoke.py)
frontend/   Vite + React + TypeScript single page; src/types.ts mirrors backend/contract.py
runpod/     Pod lifecycle: mint, sync, setup, start, status, kill
scripts/    precompute.py writes the prebuilt examples
fixtures/   A hand-shaped example at true dimensionality plus generated contrast sets
SPEC.md     The build spec, kept as the record of decisions
```

## Running it

Backend, on a machine with one 80 GB GPU:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
uv pip install --python .venv/bin/python -r backend/requirements.txt
printf 'ANTHROPIC_API_KEY=...\nDEMO_PASSWORD=...\n' > .env   # leave DEMO_PASSWORD out to run the server open
DATA_DIR=data .venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000
python3 backend/smoke.py http://localhost:8000
```

`runpod/pod.sh mint` then `runpod/launch.sh setup` and `runpod/launch.sh start` do the same on a RunPod pod.

Frontend:

```
cd frontend && npm install && npm run dev
```

Set `apiBase` in `frontend/public/config.json`. Prebuilt examples are regenerated with `scripts/ship.sh precompute`, or directly:

```
python3 scripts/precompute.py --base <api host> --decoys-variant sycophancy --deflate-variant sycophancy -- sycophancy sadness french-language sarcasm legal-language
```

Routes: `GET /health`, `POST /concept` (event stream) and `/concept.json`, `POST /fit`, `POST /axis` (event stream) and `/axis.json`, `POST /deflate`, `POST /misfires`, `POST /mine` (event stream) and `/mine.json`, `POST /chat` (event stream), `POST /score`, `GET /probe/{id}`. Shapes in `backend/contract.py`.

## Design notes

See `DESIGN.md` for the decisions and their measured consequences, and `results/NUMBERS.md` for every number.

## Reuse

The model loader, span helper, and steering hook are adapted from an emotion-probes tutorial written earlier for the same model; the server layout and degeneration flags come from an activation-steering demo. Both are credited in the source.

## License

MIT.
