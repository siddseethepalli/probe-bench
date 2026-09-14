"""End-to-end smoke test against a running server, standard library only.

    python backend/smoke.py https://<pod-id>-8000.proxy.runpod.net [concept] [--max-new-tokens N] [--skip-concept]

Waits for the model to load, fits the fixture contrast set, streams two chats (unsteered, then alpha=1.0),
scores the conversation, fetches the stored probe, then, when the generator is deployed, streams /concept
for the given concept (default: the fixture's), calls /concept.json for the now-cached set, fits it, and
streams /axis for its confound axis. Finally deflates a probe with the committed axis set under
fixtures/axis/<concept>.json, or with a small synthetic axis set built from the fixture when there is none.
Prints the numbers a user would check by hand. --skip-concept leaves the generator alone (for timing runs).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "fixtures", "example.json")
SETS_DIR = os.path.join(HERE, "..", "fixtures", "sets")
AXIS_DIR = os.path.join(HERE, "..", "fixtures", "axis")
PLATEAU = 0.01
SYNTHETIC_PAIRS = 6  # per kind


def request(base: str, method: str, path: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    # The RunPod proxy sits behind Cloudflare, which bans the default Python-urllib user agent (error 1010).
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (probe-bench smoke)"}
    if os.environ.get("PROBE_KEY"):
        headers["X-Probe-Key"] = os.environ["PROBE_KEY"]
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")


def call(base: str, method: str, path: str, body: dict | None = None) -> dict:
    with request(base, method, path, body) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_events(resp):
    """Yield (event, seconds since the request) for each `data:` block of a server-sent event stream."""
    t0, buffer = time.time(), ""
    while True:
        line = resp.readline().decode("utf-8")
        if not line:
            raise SystemExit("stream ended without a final event")
        if line.startswith("data:"):
            buffer += line[5:].strip()
            continue
        if line.strip() or not buffer:
            continue
        event, buffer = json.loads(buffer), ""
        yield event, time.time() - t0


def print_fit(fit: dict, elapsed: float) -> int:
    layer = fit["picked_layer"]
    per = fit["per_layer"]
    print(f"probe_id={fit['probe_id']} model={fit['model_id']} n_layers={fit['n_layers']} picked_layer={layer} ({elapsed:.1f}s)")
    print(f"cv_auroc[{layer}]={per['cv_auroc'][layer]:.3f}  heldout_acc[{layer}]={per['heldout_acc'][layer]:.3f}  max cv_auroc={max(per['cv_auroc']):.3f}")
    print(f"held-out AUROC at layer {layer}: shipped={per['heldout_auroc'][layer]:.3f}  matched-only={per['heldout_auroc_matched'][layer]:.3f}  background-only={per['heldout_auroc_background'][layer]:.3f}")
    print(f"confound index[{layer}]={fit['confound']['index'][layer]:+.3f}  verdict: {fit['verdict_by_layer'][layer]}")
    for g in ("explicit", "implicit", "decoys", "neutral"):
        print(f"  mean seq_z {g:9s} at layer {layer}: {fit['confound']['means'][g][layer]:+.3f}")
    print(f"stress_examples={len(fit['stress_examples'])} groups={sorted({s['group'] for s in fit['stress_examples']})}")
    st = fit.get("stability")
    if st:
        for name, at in (("accuracy pick", layer), ("best confound plateau layer", best_confound_layer(fit))):
            print(f"stability at layer {at} ({name}): split-half cosine={st['split_half_cosine'][at]:.3f} corrected={st['full_set_reliability'][at]:.3f} "
                  f"split-half AUROC={st['split_half_auroc'][at]:.3f} (n_splits={st['n_splits']})")
    return layer


def best_confound_layer(fit: dict) -> int:
    """The frontend's default view: the highest confound index among layers within PLATEAU of the best CV AUROC."""
    auroc = fit["per_layer"]["cv_auroc"]
    top = max(auroc)
    plateau = [i for i, a in enumerate(auroc) if a >= top - PLATEAU]
    return max(plateau, key=lambda i: fit["confound"]["index"][i])


def fixture_path(directory: str, concept: str) -> str:
    return os.path.join(directory, concept.strip().lower().replace(" ", "-") + ".json")


def committed_set_fit(base: str, concept: str) -> None:
    """Fit the generated set committed under fixtures/sets/<concept>.json, when there is one: real numbers, no spend."""
    path = fixture_path(SETS_DIR, concept)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        cs = json.load(f)["contrast_set"]
    t0 = time.time()
    fit = call(base, "POST", "/fit", {"concept": concept, "contrast_set": cs})
    print(f"\n=== fit on the committed generated set ({os.path.relpath(path, os.path.join(HERE, '..'))}) ===")
    print_fit(fit, time.time() - t0)


def concept_checks(base: str, concept: str) -> tuple[dict, dict] | None:
    """Stream /concept (heartbeats prove the proxy stays open), then /concept.json (cached), then fit the set.
    Returns (contrast set, fit) when the generator is deployed."""
    try:
        resp = request(base, "POST", "/concept", {"concept": concept})
    except SystemExit as e:
        if "503" in str(e):
            print(f"\n=== concept ===\ngenerator not deployed ({e}); skipping")
            return None
        raise
    print(f"\n=== concept {concept!r} (server-sent events) ===")
    with resp:
        for event, at in read_events(resp):
            if event["type"] == "status":
                print(f"  {at:6.1f}s status: {event['text']}")
                continue
            if event["type"] == "error":
                raise SystemExit(f"concept error (status {event.get('status')}): {event['error']}")
            print(f"  {at:6.1f}s concept event: cached={event['cached']} dropped={event['dropped']}")
            cs = event["contrast_set"]
            break
    print("  lists: " + ", ".join(f"{k}={len(cs[k])}" for k in ("train_pos", "train_neg", "heldout_pos", "heldout_neg", "implicit_pos", "decoys", "neutral", "background")))
    print(f"  keywords={cs['keywords']}\n  is_response_property={cs['is_response_property']}  demo_user_message={cs['demo_user_message']!r}")

    t0 = time.time()
    again = call(base, "POST", "/concept.json", {"concept": concept})
    print(f"\n=== concept.json {concept!r} ({time.time() - t0:.1f}s) ===\ncached={again['cached']} dropped={again['dropped']} train_pos={len(again['contrast_set']['train_pos'])}")

    t0 = time.time()
    fit = call(base, "POST", "/fit", {"concept": concept, "contrast_set": cs})
    print(f"\n=== fit on the generated set ===")
    print_fit(fit, time.time() - t0)
    return cs, fit


def axis_checks(base: str, concept: str, cs: dict, fit: dict) -> None:
    """Stream /axis for the generated set, then /axis.json (cached), then deflate the set's probe with the result."""
    try:
        resp = request(base, "POST", "/axis", {"concept": concept, "contrast_set": cs})
    except SystemExit as e:
        if "503" in str(e):
            print(f"\n=== axis ===\naxis generator not deployed ({e}); skipping")
            return
        raise
    print(f"\n=== axis {concept!r} (server-sent events) ===")
    with resp:
        for event, at in read_events(resp):
            if event["type"] == "status":
                print(f"  {at:6.1f}s status: {event['text']}")
                continue
            if event["type"] == "error":
                raise SystemExit(f"axis error (status {event.get('status')}): {event['error']}")
            print(f"  {at:6.1f}s axis event: cached={event['cached']} dropped={event['dropped']}")
            axis_set = event["axis_set"]
            break
    print(f"  axis={axis_set['axis']!r} absent_pairs={len(axis_set['absent_pairs'])} present_pairs={len(axis_set['present_pairs'])}")
    for kind in ("absent_pairs", "present_pairs"):
        if axis_set[kind]:
            print(f"  {kind}[0]: high={axis_set[kind][0]['high'][:70]!r}\n{'':17s}low={axis_set[kind][0]['low'][:70]!r}")

    t0 = time.time()
    again = call(base, "POST", "/axis.json", {"concept": concept, "contrast_set": cs})
    print(f"\n=== axis.json {concept!r} ({time.time() - t0:.1f}s) ===\ncached={again['cached']} dropped={again['dropped']}")
    run_deflate(base, "deflate the generated set's probe with the generated axis set", cs, fit, axis_set)


def synthetic_axis_set(cs: dict) -> dict:
    """A stand-in axis set from the fixture's own lists, enough to exercise /deflate: decoys (keywords, warm,
    concept absent) over matched negatives for the absent pairs, explicit positives over implicit positives
    (concept present, keywords and warmth flipped) for the present pairs. Each pair keeps the high text's context."""
    keywords = [k.lower() for k in cs["keywords"]]
    explicit = [e for e in cs["train_pos"] if any(k in e["text"].lower() for k in keywords)]

    def pairs(highs: list[dict], lows: list[dict]) -> list[dict]:
        return [{"high": h["text"], "low": l["text"], "context": h.get("context")} for h, l in list(zip(highs, lows))[:SYNTHETIC_PAIRS]]

    return {
        "axis": "keyword warmth (synthetic)",
        "absent_pairs": pairs(cs["decoys"], cs["train_neg"]),
        "present_pairs": pairs(explicit, cs["implicit_pos"]),
    }


def score_reply(base: str, probe_id: str, cs: dict, example: dict, layer: int) -> float:
    """seq_z of ``example`` read as an assistant reply (to its own context, or the demo message) under a probe."""
    messages = [{"role": "user", "content": example.get("context") or cs["demo_user_message"]}, {"role": "assistant", "content": example["text"]}]
    return call(base, "POST", "/score", {"probe_id": probe_id, "messages": messages})["turns"][-1]["seq_z"][layer]


def run_deflate(base: str, label: str, cs: dict, fit: dict, axis_set: dict) -> None:
    """POST /deflate and print the evidence at the two layers the page shows: the best confound layer on the
    plateau (the default view) and the accuracy pick, both from the probe before deflation. Then read one panel
    decoy and one implicit positive through /score under both probes, and stream a short /chat on the new one."""
    t0 = time.time()
    out = call(base, "POST", "/deflate", {"probe_id": fit["probe_id"], "axis_set": axis_set})
    after = out["fit"]
    print(f"\n=== {label} ({time.time() - t0:.1f}s) ===")
    print(f"axis={out['axis']!r} pairs={len(axis_set['absent_pairs'])}+{len(axis_set['present_pairs'])}")
    print(f"deflated probe_id={after['probe_id']} deflated_axis={after['deflated_axis']!r} picked_layer={after['picked_layer']} "
          f"best confound plateau layer={best_confound_layer(after)} (before: picked_layer={fit['picked_layer']} best confound plateau layer={best_confound_layer(fit)})")
    for name, layer in (("best confound plateau layer", best_confound_layer(fit)), ("accuracy pick", fit["picked_layer"])):
        print(f"at layer {layer} ({name}):")
        print(f"  axis CV AUROC          before={out['axis_cv_auroc_before'][layer]:.3f}  after={out['axis_cv_auroc_after'][layer]:.3f}")
        print(f"  concept held-out AUROC before={out['heldout_auroc_before'][layer]:.3f}  after={out['heldout_auroc_after'][layer]:.3f}")
        print(f"  confound index         before={out['confound_index_before'][layer]:+.3f}  after={out['confound_index_after'][layer]:+.3f}  verdict after: {after['verdict_by_layer'][layer]}")
        print("  mean seq_z after: " + "  ".join(f"{g}={after['confound']['means'][g][layer]:+.3f}" for g in ("explicit", "implicit", "decoys", "neutral")))
        st = after.get("stability")
        if st:
            print(f"  stability after: split-half cosine={st['split_half_cosine'][layer]:.3f} corrected={st['full_set_reliability'][layer]:.3f} split-half AUROC={st['split_half_auroc'][layer]:.3f}")

    layer = best_confound_layer(fit)
    panel = cs["decoys"][6:] if len(cs["decoys"]) >= 12 else cs["decoys"]
    for name, lst in (("panel decoy", panel), ("implicit positive", cs["implicit_pos"])):
        if lst:
            before, after_z = (score_reply(base, pid, cs, lst[0], layer) for pid in (fit["probe_id"], after["probe_id"]))
            print(f"/score {name} at layer {layer}: before={before:+.3f} after={after_z:+.3f}  {lst[0]['text'][:60]!r}")
    body = {"probe_id": after["probe_id"], "system_prompt": cs["system_prompts"][0], "messages": [{"role": "user", "content": cs["demo_user_message"]}], "alpha": 0.0, "max_new_tokens": 40, "temperature": 0.7, "seed": 0}
    scored, _, t0 = stream_chat(base, body)
    reply = scored["turns"][-1]
    print(f"/chat on the deflated probe ({time.time() - t0:.1f}s): {len(reply['tokens'])} tokens, reply seq_z at layer {layer} = {reply['seq_z'][layer]:+.3f}, flags={scored['flags']}")


def deflate_checks(base: str, concept: str, fixture_cs: dict, fixture_fit: dict) -> None:
    """Deflate with the committed axis set for the concept when both fixtures exist, else with a synthetic
    axis set on the fixture probe, so the route runs either way."""
    set_path, axis_path = fixture_path(SETS_DIR, concept), fixture_path(AXIS_DIR, concept)
    if os.path.exists(set_path) and os.path.exists(axis_path):
        with open(set_path, encoding="utf-8") as f:
            cs = json.load(f)["contrast_set"]
        with open(axis_path, encoding="utf-8") as f:
            axis_set = json.load(f)["axis_set"]
        fit = call(base, "POST", "/fit", {"concept": concept, "contrast_set": cs})
        run_deflate(base, f"deflate with the committed axis set ({os.path.relpath(axis_path, os.path.join(HERE, '..'))})", cs, fit, axis_set)
        return
    run_deflate(base, "deflate the fixture probe with a synthetic axis set", fixture_cs, fixture_fit, synthetic_axis_set(fixture_cs))


def misfire_checks(base: str, cs: dict, fit: dict) -> None:
    """POST /misfires at the accuracy pick and the best confound plateau layer: the top tokens the direction
    fires on when the concept is absent, and the seeds; then /mine with the viewing layer's seeds when the
    decoy miner is deployed."""
    seeds = []
    for name, layer in (("accuracy pick", fit["picked_layer"]), ("best confound plateau layer", best_confound_layer(fit))):
        t0 = time.time()
        out = call(base, "POST", "/misfires", {"probe_id": fit["probe_id"], "layer": layer, "top_k": 10})
        print(f"\n=== misfires at layer {out['layer']} ({name}) ({time.time() - t0:.1f}s) ===")
        for m in out["misfires"]:
            print(f"  z={m['z']:+6.2f}  {m['token']!r:14s} {m['group']:10s} {m['text'][:60]!r}")
        counts = {g: sum(m["group"] == g for m in out["misfires"]) for g in ("matched", "background", "neutral", "decoys")}
        print(f"  groups in the top {len(out['misfires'])}: {counts}")
        print(f"  seeds: {out['seeds']}")
        seeds = out["seeds"]
    try:
        resp = request(base, "POST", "/mine", {"concept": cs["concept"], "contrast_set": cs, "seeds": seeds})
    except SystemExit as e:
        if "503" in str(e):
            print(f"\n=== mine ===\ndecoy miner not deployed ({e}); skipping")
            return
        raise
    print(f"\n=== mine {cs['concept']!r} from {len(seeds)} seeds (server-sent events) ===")
    with resp:
        for event, at in read_events(resp):
            if event["type"] == "status":
                print(f"  {at:6.1f}s status: {event['text']}")
                continue
            if event["type"] == "error":
                raise SystemExit(f"mine error (status {event.get('status')}): {event['error']}")
            print(f"  {at:6.1f}s mined event: cached={event['cached']} dropped={event['dropped']} decoys={len(event['decoys'])}")
            for e in event["decoys"][:3]:
                print(f"    {e['text'][:90]!r}")
            break
    t0 = time.time()
    again = call(base, "POST", "/mine.json", {"concept": cs["concept"], "contrast_set": cs, "seeds": seeds})
    print(f"\n=== mine.json ({time.time() - t0:.1f}s) ===\ncached={again['cached']} dropped={again['dropped']} decoys={len(again['decoys'])}")


def stream_chat(base: str, body: dict) -> tuple[dict, list[float], float]:
    """POST /chat and consume the SSE stream. Returns (scored event, timestamps of non-empty token events, t0)."""
    t0 = time.time()
    stamps: list[float] = []
    with request(base, "POST", "/chat", body) as resp:
        for event, _ in read_events(resp):
            if event["type"] == "token":
                if event["text"]:
                    stamps.append(time.time())
            elif event["type"] == "scored":
                return event, stamps, t0
            else:
                raise SystemExit(f"chat error event: {event.get('error')}")


def report_chat(label: str, scored: dict, stamps: list[float], t0: float, layer: int) -> None:
    reply_turn = scored["turns"][-1]
    n_reply = len(reply_turn["tokens"])
    ttft = stamps[0] - t0 if stamps else float("nan")
    decode_s = stamps[-1] - stamps[0] if len(stamps) > 1 else float("nan")
    rate = (n_reply - 1) / decode_s if decode_s and decode_s > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"reply ({n_reply} tokens, hit_length_cap={scored['hit_length_cap']}, flags={scored['flags']}):")
    print("  " + scored["reply"].replace("\n", "\n  "))
    print(f"per-turn seq_z at layer {layer}: " + ", ".join(f"{t['role']}={t['seq_z'][layer]:+.2f}" for t in scored["turns"]))
    print(f"conversation_tokens={scored['conversation_tokens']}  turns={[t['role'] for t in scored['turns']]}")
    print(f"time to first token {ttft:.2f}s | decode {decode_s:.2f}s for {n_reply} tokens -> {rate:.1f} tok/s | total {time.time() - t0:.1f}s")
    z_shape = (len(reply_turn["z"]), len(reply_turn["z"][0]) if reply_turn["z"] else 0)
    print(f"reply z shape {z_shape}; first tokens: {reply_turn['tokens'][:6]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe Bench end-to-end smoke test")
    ap.add_argument("base_url")
    ap.add_argument("concept", nargs="?", help="concept for the /concept checks (default: the fixture's)")
    ap.add_argument("--max-new-tokens", type=int, default=200, help="reply length for the chat checks")
    ap.add_argument("--skip-concept", action="store_true", help="skip the /concept, /concept.json and generated-set fit")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    with open(FIXTURE, encoding="utf-8") as f:
        fixture = json.load(f)
    cs = fixture["contrast_set"]
    concept = args.concept or cs["concept"]

    deadline = time.time() + 900
    while True:
        health = call(base, "GET", "/health")
        print("health:", health)
        if health["model_loaded"]:
            break
        if time.time() > deadline:
            raise SystemExit("model did not load in 15 minutes")
        time.sleep(10)

    t0 = time.time()
    fit = call(base, "POST", "/fit", {"concept": cs["concept"], "contrast_set": cs})
    print("\n=== fit on the fixture set ===")
    layer = print_fit(fit, time.time() - t0)

    messages = [{"role": "user", "content": cs["demo_user_message"]}]
    body = {"probe_id": fit["probe_id"], "system_prompt": cs["system_prompts"][0], "messages": messages, "alpha": 0.0, "max_new_tokens": args.max_new_tokens, "temperature": 0.7, "seed": 0}
    scored, stamps, t0 = stream_chat(base, body)
    report_chat("chat alpha=0", scored, stamps, t0, layer)

    scored_steered, stamps, t0 = stream_chat(base, {**body, "alpha": 1.0})
    report_chat("chat alpha=1.0", scored_steered, stamps, t0, layer)

    conversation = messages + [{"role": "assistant", "content": scored["reply"]}]
    t0 = time.time()
    score = call(base, "POST", "/score", {"probe_id": fit["probe_id"], "system_prompt": cs["system_prompts"][0], "messages": conversation})
    print(f"\n=== score ({time.time() - t0:.1f}s) ===")
    print(f"conversation_tokens={score['conversation_tokens']} turns={[t['role'] for t in score['turns']]}")
    print(f"per-turn seq_z at layer {layer}: " + ", ".join(f"{t['role']}={t['seq_z'][layer]:+.2f}" for t in score["turns"]))
    chat_reply_z = scored["turns"][-1]["seq_z"][layer]
    print(f"reply seq_z from /chat {chat_reply_z:+.3f} vs /score {score['turns'][-1]['seq_z'][layer]:+.3f} (bf16 batch-shape noise expected)")

    info = call(base, "GET", f"/probe/{fit['probe_id']}")
    print(f"\n=== probe ===\nprobe_id={info['probe_id']} concept={info['concept']} picked_layer={info['picked_layer']} contrast_set lists: "
          + ", ".join(f"{k}={len(info['contrast_set'][k])}" for k in ("train_pos", "train_neg", "heldout_pos", "heldout_neg", "implicit_pos", "decoys", "neutral")))
    misfire_checks(base, cs, fit)
    if not args.skip_concept:
        committed_set_fit(base, concept)
        generated = concept_checks(base, concept)
        if generated is not None:
            axis_checks(base, concept, *generated)
    deflate_checks(base, concept, cs, fit)
    print("\nhealth:", call(base, "GET", "/health"))
    print("\nsmoke passed")


if __name__ == "__main__":
    main()
