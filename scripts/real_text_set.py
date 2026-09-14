"""Build a prebuilt example whose whole contrast set is real text: a sadness probe fit on GoEmotions
comments, with real stress sets, so it can be compared with the probe fit on generated text.

    PROBE_KEY=... python3 scripts/real_text_set.py --base <api host>

Positives are comments labeled sadness (half with a sadness keyword, half without); negatives are
comments with any other label and no keyword; decoys are keyword-bearing comments not labeled sad;
neutral is neutral-labeled text. Keywords, system prompts, and the demo message come from the
generated sadness set so the canned conversations are comparable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import random
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from precompute import build_example  # noqa: E402

RAW = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"


def fetch(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "probe-bench-realtext/1.0"}), timeout=120) as r:
        return r.read().decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--examples", default="frontend/public/examples")
    args = ap.parse_args()
    random.seed(3)
    src = json.loads(pathlib.Path(args.examples, "sadness.json").read_text())
    gen = src["contrast_set"]
    keywords = [k.lower() for k in gen["keywords"] if len(k) >= 3]
    kw = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")", re.I)
    labels = fetch(RAW + "emotions.txt").split()
    sad, neutral = labels.index("sadness"), labels.index("neutral")
    rows = list(csv.reader(io.StringIO(fetch(RAW + "train.tsv")), delimiter="\t"))
    good = [(t, {int(x) for x in ids.split(",")}) for t, ids, _ in rows if 8 <= len(t.split()) <= 40 and "[NAME]" not in t]
    sad_kw = [t for t, ids in good if ids == {sad} and kw.search(t)]
    sad_nokw = [t for t, ids in good if ids == {sad} and not kw.search(t)]
    other = [t for t, ids in good if sad not in ids and neutral not in ids and not kw.search(t)]
    neutral_nokw = [t for t, ids in good if ids == {neutral} and not kw.search(t)]
    decoy_pool = [t for t, ids in good if sad not in ids and kw.search(t)]
    random.shuffle(sad_kw); random.shuffle(sad_nokw); random.shuffle(other); random.shuffle(neutral_nokw); random.shuffle(decoy_pool)
    ex = lambda texts: [{"text": t, "context": None} for t in texts]
    cs = {
        "concept": "sadness", "is_response_property": False, "keywords": gen["keywords"],
        "train_pos": ex(sad_kw[:16] + sad_nokw[:16]),
        "train_neg": ex(other[:32]),
        "heldout_pos": ex(sad_kw[16:19] + sad_nokw[16:19]),
        "heldout_neg": ex(other[32:38]),
        "implicit_pos": ex(sad_nokw[19:25]),
        "decoys": ex(decoy_pool[:12]),
        "neutral": ex(neutral_nokw[:6]),
        "confound_axis": gen.get("confound_axis"),
        "background": ex(other[38:62]),
        "system_prompts": gen["system_prompts"],
        "demo_user_message": gen["demo_user_message"],
    }
    print({k: len(v) for k, v in cs.items() if isinstance(v, list)})
    example = build_example(args.base.rstrip("/"), "sadness-real-fit", "sadness", cs)
    example["title"] = "Sadness, fit on real text"
    example["blurb"] = "The same concept and keywords, but every example is a real Reddit comment from GoEmotions: positives labeled sad, negatives with other labels, real keyword decoys. Compare with the generated probe."
    out = pathlib.Path(args.examples, "sadness-real-fit.json"); out.write_text(json.dumps(example))
    idx_path = pathlib.Path(args.examples, "index.json"); idx = json.loads(idx_path.read_text())
    idx = [e for e in idx if e["slug"] != "sadness-real-fit"]
    pos = next((i for i, e in enumerate(idx) if e["slug"] == "sadness-real-text"), len(idx) - 1) + 1
    idx.insert(pos, {"slug": example["slug"], "title": example["title"], "blurb": example["blurb"]})
    idx_path.write_text(json.dumps(idx, indent=1))
    fit = example["fit"]; l = fit["picked_layer"]
    print(f"wrote {out}; picked {l}, index {fit['confound']['index'][l]:+.2f} ({fit['verdict_by_layer'][l]}), stability {fit['stability']['full_set_reliability'][l]:.2f}")


if __name__ == "__main__":
    main()
