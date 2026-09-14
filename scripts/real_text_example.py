"""Build a prebuilt example whose stress sets are real text: GoEmotions comments scored under the
generated sadness probe. Everything else (contrast set, fit, canned conversations) is copied from the
generated sadness example, so the two chips differ only in what the confound panel is judging.

    python3 scripts/real_text_example.py --base <api host> --n 12
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import os
import random
import re
import urllib.request

RAW = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
HEADERS = {"Content-Type": "application/json", "User-Agent": "probe-bench-realtext/1.0"}
if os.environ.get("PROBE_KEY"):  # the API's access password; sent to the API only, never to the dataset host
    HEADERS["X-Probe-Key"] = os.environ["PROBE_KEY"]
VERDICTS = [(1.0, "tracks the concept"), (0.3, "mixed"), (-1e9, "tracks the words")]


def fetch(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=120) as r:
        return r.read().decode()


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--examples", default="frontend/public/examples")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    random.seed(1)
    src = json.loads(pathlib.Path(args.examples, "sadness.json").read_text())
    cs, fit = src["contrast_set"], src["fit"]
    keywords = [k.lower() for k in cs["keywords"] if len(k) >= 3]
    kw = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")", re.I)
    labels = fetch(RAW + "emotions.txt").split()
    sad, neutral = labels.index("sadness"), labels.index("neutral")
    rows = list(csv.reader(io.StringIO(fetch(RAW + "train.tsv")), delimiter="\t"))
    good = [(t, {int(x) for x in ids.split(",")}) for t, ids, _ in rows if 6 <= len(t.split()) <= 40 and "[NAME]" not in t]
    pools = {
        "explicit": [t for t, ids in good if ids == {sad} and kw.search(t)],
        "implicit": [t for t, ids in good if ids == {sad} and not kw.search(t)],
        "decoys": [t for t, ids in good if sad not in ids and kw.search(t)],
        "neutral": [t for t, ids in good if ids == {neutral} and not kw.search(t)],
    }
    print({k: len(v) for k, v in pools.items()})
    layers = fit["n_layers"]
    stress, means = [], {g: [0.0] * layers for g in pools}
    for group, pool in pools.items():
        picks = random.sample(pool, args.n)
        for text in picks:
            r = post(base, "/score", {"probe_id": fit["probe_id"], "system_prompt": "", "messages": [{"role": "user", "content": text}]})
            z = r["turns"][0]["seq_z"]
            stress.append({"text": text, "context": None, "group": group, "seq_z": [round(v, 3) for v in z]})
            means[group] = [m + v / args.n for m, v in zip(means[group], z)]
        print(f"  {group}: {args.n} scored, mean at layer {fit['picked_layer']} = {means[group][fit['picked_layer']]:+.2f}")
    index = [round(means["implicit"][l] - means["decoys"][l], 3) for l in range(layers)]
    verdict = [next(v for thr, v in VERDICTS if i > thr or (thr == 1.0 and i > 1.0)) for i in index]
    verdict = ["tracks the concept" if i > 1.0 else ("mixed" if i >= 0.3 else "tracks the words") for i in index]
    new_fit = dict(fit)
    new_fit["confound"] = {"index": index, "means": {g: [round(v, 3) for v in means[g]] for g in means}}
    new_fit["verdict_by_layer"] = verdict
    new_fit["stress_examples"] = stress
    example = {
        "slug": "sadness-real-text",
        "title": "Sadness, judged on real text",
        "blurb": "The same generated probe, but the stress sets are real Reddit comments from GoEmotions: sad comments with and without the keywords, keyword-bearing comments that are not sad, and neutral ones.",
        "contrast_set": cs, "fit": new_fit, "conversations": src["conversations"],
    }
    out = pathlib.Path(args.examples, "sadness-real-text.json"); out.write_text(json.dumps(example))
    idx_path = pathlib.Path(args.examples, "index.json"); idx = json.loads(idx_path.read_text())
    idx = [e for e in idx if e["slug"] != "sadness-real-text"]
    pos = next((i for i, e in enumerate(idx) if e["slug"] == "sadness"), len(idx) - 1) + 1
    idx.insert(pos, {"slug": example["slug"], "title": example["title"], "blurb": example["blurb"]})
    idx_path.write_text(json.dumps(idx, indent=1))
    l = fit["picked_layer"]
    print(f"wrote {out}; real-text index at layer {l}: {index[l]:+.2f} ({verdict[l]}) vs generated {fit['confound']['index'][l]:+.2f}")


if __name__ == "__main__":
    main()
