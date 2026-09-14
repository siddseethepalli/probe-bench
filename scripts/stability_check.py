"""Split-half stability of the fitted direction: is the contrast set big enough?

    python3 scripts/stability_check.py --base <api host> --seeds 3 sycophancy sadness

For each seed the training pools (train_pos, train_neg, background, neutral) are split in half at random.
Each half is fit through /fit with the OTHER half's positives and negatives passed as the held-out set,
so the returned held-out AUROC is the split-half accuracy (direction from A, tested on B). The stored
direction files are then pulled off the pod (runpod/launch.sh run) and the cosine between the two
half-directions is computed per layer. Spearman-Brown corrects the split-half figure toward the full
set: r_full = 2r / (1 + r).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import urllib.request

import numpy as np

HEADERS = {"Content-Type": "application/json", "User-Agent": "probe-bench-stability/1.0"}
if os.environ.get("PROBE_KEY"):  # the API's access password, read from the environment only
    HEADERS["X-Probe-Key"] = os.environ["PROBE_KEY"]


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def halves(items: list, rng: random.Random) -> tuple[list, list]:
    order = items[:]
    rng.shuffle(order)
    mid = len(order) // 2
    return order[:mid], order[mid:]


def fetch_dhat(probe_ids: list[str]) -> dict[str, np.ndarray]:
    cmd = "cd /workspace/data/probes && python3 -c \"import json,sys; print(json.dumps({p: json.load(open(p + '.json'))['dhat_f16_b64'] for p in sys.argv[1:]}))\" " + " ".join(probe_ids)
    out = subprocess.run(["runpod/launch.sh", "run", cmd], capture_output=True, text=True, check=True).stdout
    raw = json.loads(out.strip().splitlines()[-1])
    result = {}
    for pid, b64 in raw.items():
        flat = np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype(np.float32)
        result[pid] = flat
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("slugs", nargs="+")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    for slug in args.slugs:
        cs = json.load(open(f"fixtures/sets/{slug}.json"))["contrast_set"]
        full = post(base, "/fit", {"concept": cs["concept"], "contrast_set": cs})
        L = full["n_layers"]
        cv = full["per_layer"]["cv_auroc"]
        floor = L // 4
        plateau = [l for l in range(floor, L) if cv[l] >= max(cv[floor:]) - 0.01]
        view = max(plateau, key=lambda l: full["confound"]["index"][l])
        pick = full["picked_layer"]
        print(f"== {slug}: full set picked {pick}, viewing layer {view}, index {full['confound']['index'][view]:+.2f}")
        cos_all, auroc_all, idx_pairs = [], [], []
        pids = []
        fits = []
        for seed in range(args.seeds):
            rng = random.Random(seed)
            pos_a, pos_b = halves(cs["train_pos"], rng)
            neg_a, neg_b = halves(cs["train_neg"], rng)
            bg_a, bg_b = halves(cs["background"], rng)
            nt_a, nt_b = halves(cs["neutral"], rng)
            for mine, other in [((pos_a, neg_a, bg_a, nt_a), (pos_b, neg_b)), ((pos_b, neg_b, bg_b, nt_b), (pos_a, neg_a))]:
                half = dict(cs)
                half["train_pos"], half["train_neg"], half["background"], half["neutral"] = mine
                half["heldout_pos"], half["heldout_neg"] = other
                fit = post(base, "/fit", {"concept": cs["concept"], "contrast_set": half})
                fits.append(fit)
                pids.append(fit["probe_id"])
        dhats = fetch_dhat(list(dict.fromkeys(pids)))
        for i in range(0, len(fits), 2):
            a, b = fits[i], fits[i + 1]
            da = dhats[a["probe_id"]].reshape(L, -1)
            db = dhats[b["probe_id"]].reshape(L, -1)
            cos = np.sum(da * db, axis=1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1) + 1e-8)
            cos_all.append(cos)
            auroc_all.append((np.array(a["per_layer"]["heldout_auroc"]) + np.array(b["per_layer"]["heldout_auroc"])) / 2)
            idx_pairs.append((a["confound"]["index"], b["confound"]["index"]))
        cos_mean = np.mean(cos_all, axis=0)
        auroc_mean = np.mean(auroc_all, axis=0)
        for name, layer in [("viewing layer", view), ("accuracy pick", pick)]:
            c = cos_mean[layer]
            sb = 2 * c / (1 + c)
            idx_a = np.mean([p[0][layer] for p in idx_pairs])
            idx_b = np.mean([p[1][layer] for p in idx_pairs])
            idx_sd = np.std([p[0][layer] for p in idx_pairs] + [p[1][layer] for p in idx_pairs])
            print(f"   {name} {layer:>2}: split-half cosine {c:.3f} (Spearman-Brown to full n: {sb:.3f}) | split-half AUROC {auroc_mean[layer]:.3f} | "
                  f"half-fit confound index {idx_a:+.2f} / {idx_b:+.2f} (sd {idx_sd:.2f})")
        print("   cosine by layer:", " ".join(f"{l}:{cos_mean[l]:.2f}" for l in range(0, L, 8)))


if __name__ == "__main__":
    main()
