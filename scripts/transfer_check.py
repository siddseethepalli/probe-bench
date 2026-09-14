"""Transfer check on real text: does a sadness probe fit on generated examples transfer to Reddit comments?

    python scripts/transfer_check.py --base https://<api host> [--n 40]

Uses GoEmotions (Google, Apache-2.0): comments labeled sadness (real positives), comments that contain a
sadness keyword but are not labeled sadness (real keyword decoys), and comments labeled neutral with no
keyword (real background). Each text is scored as a user turn, the same framing the probe was fit in.
Reports AUROC and group means raw and after per-corpus centering (subtracting the neutral mean).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import urllib.request

RAW = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
HEADERS = {"Content-Type": "application/json", "User-Agent": "probe-bench-transfer/1.0"}
if os.environ.get("PROBE_KEY"):  # the API's access password; sent to the API only, never to the dataset host
    HEADERS["X-Probe-Key"] = os.environ["PROBE_KEY"]


def fetch(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=120) as r:
        return r.read().decode()


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--set", default="fixtures/sets/sadness.json")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    random.seed(0)

    cs = json.load(open(args.set))["contrast_set"]
    keywords = [k.lower() for k in cs["keywords"] if len(k) >= 3]
    kw = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")", re.I)
    labels = fetch(RAW + "emotions.txt").split()
    sad, neutral = labels.index("sadness"), labels.index("neutral")
    rows = list(csv.reader(io.StringIO(fetch(RAW + "train.tsv")), delimiter="\t"))
    good = [(t, {int(x) for x in ids.split(",")}) for t, ids, _ in rows if 6 <= len(t.split()) <= 40 and "[NAME]" not in t]
    pos = [t for t, ids in good if ids == {sad}]
    decoys = [t for t, ids in good if sad not in ids and kw.search(t)]
    background = [t for t, ids in good if ids == {neutral} and not kw.search(t)]
    groups = {"real sad": random.sample(pos, args.n), "real keyword decoys": random.sample(decoys, args.n),
              "real neutral": random.sample(background, args.n)}
    print(f"corpus pools: sad {len(pos)}, keyword decoys {len(decoys)}, neutral {len(background)}; sampling {args.n} each")

    fit = post(base, "/fit", {"concept": cs["concept"], "contrast_set": cs})
    layer = fit["picked_layer"]
    print(f"probe {fit['probe_id']} picked layer {layer}; generated-set confound index {fit['confound']['index'][layer]:+.2f}")

    scores: dict[str, list[float]] = {}
    for name, texts in groups.items():
        scores[name] = []
        for t in texts:
            r = post(base, "/score", {"probe_id": fit["probe_id"], "system_prompt": "", "messages": [{"role": "user", "content": t}]})
            scores[name].append(r["turns"][0]["seq_z"][layer])
    center = sum(scores["real neutral"]) / len(scores["real neutral"])
    print(f"\n{'group':<22}{'mean z':>8}{'centered':>10}")
    for name, zs in scores.items():
        m = sum(zs) / len(zs)
        print(f"{name:<22}{m:>8.2f}{m - center:>10.2f}")
    print(f"\nAUROC sad vs neutral:        {auroc(scores['real sad'], scores['real neutral']):.3f}")
    print(f"AUROC sad vs keyword decoys: {auroc(scores['real sad'], scores['real keyword decoys']):.3f}")
    print(f"real-text confound index (sad minus decoys, centered): {(sum(scores['real sad']) - sum(scores['real keyword decoys'])) / args.n:+.2f}")
    print("(centering shifts every group equally, so AUROC and the index are unchanged by it; it matters only for thresholds)")


if __name__ == "__main__":
    main()
