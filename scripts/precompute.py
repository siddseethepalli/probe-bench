"""Precompute prebuilt examples against a running backend and write them as static JSON.

    python scripts/precompute.py --base https://<pod>-8000.proxy.runpod.net \
        --out frontend/public/examples sycophancy sadness french-language sarcasm legal-language

Each concept becomes frontend/public/examples/<slug>.json (a StaticExample) with one canned
conversation per generated system prompt. Concepts listed with --decoys-variant are saved a
second time after the first six decoys are appended to the training negatives, with slug
<slug>-decoys, so both states of the verdict can be linked. index.json is rewritten last.

Standard library only, so it runs from any laptop without the backend's dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

TITLES = {
    "sycophancy": ("Sycophancy", "Agreeing with the user against the facts. The naive direction is expected to track warm phrasing."),
    "sycophancy-decoys": ("Sycophancy, after adding decoys", "The same probe refit with six warm corrections added as negatives."),
    "sycophancy-deflated": ("Sycophancy, warmth projected out", "The naive probe refit on activations with a warmth direction removed, fit from pairs that hold the concept fixed."),
    "sadness": ("Sadness", "A concept with an obvious keyword trap."),
    "french-language": ("French language", "A concept the model should represent cleanly."),
    "sarcasm": ("Sarcasm", "An abstract concept; the verdict may read as mixed."),
    "legal-language": ("Legal language", "A register rather than a topic."),
}
CONCEPT_TEXT = {"french-language": "French language", "legal-language": "legal language"}
# Cloudflare in front of the pod proxy rejects Python's default user agent with a 403.
HEADERS = {"Content-Type": "application/json", "User-Agent": "probe-bench-precompute/1.0"}
if os.environ.get("PROBE_KEY"):  # the API's access password, read from the environment only
    HEADERS["X-Probe-Key"] = os.environ["PROBE_KEY"]


def slugify(concept: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", concept.lower()).strip("-")


def post_json(base: str, path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{path} failed: {err.code} {err.read().decode()[:300]}")


def stream_final(base: str, path: str, body: dict, final_type: str, timeout: float = 600) -> dict:
    """Consume an event stream and return the final event of the given type."""
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = ""
        while True:
            chunk = resp.read(4096).decode()
            if not chunk:
                break
            buffer += chunk
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                for line in event.splitlines():
                    if line.startswith("data:"):
                        payload = json.loads(line[5:].strip())
                        if payload["type"] == final_type:
                            payload.pop("type")
                            return payload
                        if payload["type"] == "error":
                            raise SystemExit(f"{path} error: {payload['error']}")
    raise SystemExit(f"{path} stream ended without a {final_type} event")


def deflate_fit(base: str, fit: dict, axis_set: dict) -> dict:
    resp = post_json(base, "/deflate", {"probe_id": fit["probe_id"], "axis_set": axis_set})
    layer = resp["fit"]["picked_layer"]
    print(f"  deflate ok ({resp['axis']}): axis separability {resp['axis_cv_auroc_before'][layer]:.2f} -> {resp['axis_cv_auroc_after'][layer]:.2f}, "
          f"confound {resp['confound_index_before'][layer]:+.2f} -> {resp['confound_index_after'][layer]:+.2f} at layer {layer}")
    return resp["fit"]


def build_example(base: str, slug: str, concept: str, contrast_set: dict | None, label_suffix: str = "", axis_set: dict | None = None) -> dict:
    if contrast_set is None:
        concept_resp = stream_final(base, "/concept", {"concept": concept}, "concept")
        contrast_set = concept_resp["contrast_set"]
        print(f"  concept ok (cached={concept_resp['cached']}, dropped={concept_resp['dropped']})")
    fit = post_json(base, "/fit", {"concept": concept, "contrast_set": contrast_set})
    if axis_set is not None:
        fit = deflate_fit(base, fit, axis_set)
    layer = fit["picked_layer"]
    print(f"  fit ok: layer {layer}, cv_auroc {fit['per_layer']['cv_auroc'][layer]:.3f}, "
          f"confound {fit['confound']['index'][layer]:.2f}, verdict {fit['verdict_by_layer'][layer]}")
    conversations = []
    for label, system_prompt in zip(["raise", "lower"], contrast_set["system_prompts"]):
        started = time.time()
        scored = stream_final(base, "/chat", {
            "probe_id": fit["probe_id"], "system_prompt": system_prompt,
            "messages": [{"role": "user", "content": contrast_set["demo_user_message"]}],
            "alpha": 0.0, "layer": None, "max_new_tokens": 400, "temperature": 0.7, "seed": 0,
        }, "scored")
        reply_z = scored["turns"][-1]["seq_z"][layer]
        print(f"  chat '{label}' ok in {time.time() - started:.0f}s: reply seq_z {reply_z:.2f} at layer {layer}")
        conversations.append({"label": label + label_suffix, "system_prompt": system_prompt,
                              "user_message": contrast_set["demo_user_message"], "scored": scored})
    title, blurb = TITLES.get(slug, (concept.capitalize(), ""))
    return {"slug": slug, "title": title, "blurb": blurb, "contrast_set": contrast_set, "fit": fit, "conversations": conversations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="frontend/public/examples")
    ap.add_argument("--decoys-variant", nargs="*", default=["sycophancy"])
    ap.add_argument("--deflate-variant", nargs="*", default=["sycophancy"], help="slugs with a fixtures/axis/<slug>.json to project out")
    ap.add_argument("slugs", nargs="+")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    written = []
    for slug in args.slugs:
        concept = CONCEPT_TEXT.get(slug, slug.replace("-", " "))
        print(f"== {slug} ({concept})")
        example = build_example(base, slug, concept, None)
        (out / f"{slug}.json").write_text(json.dumps(example))
        written.append(example)
        if slug in args.decoys_variant:
            cs = json.loads(json.dumps(example["contrast_set"]))
            cs["train_neg"] = cs["train_neg"] + cs["decoys"][:6]
            print(f"== {slug}-decoys (six decoys added as negatives)")
            variant = build_example(base, f"{slug}-decoys", concept, cs)
            (out / f"{slug}-decoys.json").write_text(json.dumps(variant))
            written.append(variant)
        axis_path = pathlib.Path(f"fixtures/axis/{slug}.json")
        if slug in args.deflate_variant and axis_path.exists():
            axis_set = json.loads(axis_path.read_text())["axis_set"]
            print(f"== {slug}-deflated ({axis_set['axis']} projected out)")
            variant = build_example(base, f"{slug}-deflated", concept, example["contrast_set"], axis_set=axis_set)
            (out / f"{slug}-deflated.json").write_text(json.dumps(variant))
            written.append(variant)
    existing = {}
    index_path = out / "index.json"
    if index_path.exists():
        existing = {e["slug"]: e for e in json.loads(index_path.read_text())}
    for ex in written:
        existing[ex["slug"]] = {"slug": ex["slug"], "title": ex["title"], "blurb": ex["blurb"]}
    index_path.write_text(json.dumps(list(existing.values()), indent=1))
    print(f"wrote {len(written)} examples; index has {len(existing)} entries")


if __name__ == "__main__":
    sys.exit(main())
