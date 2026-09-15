"""Mean-difference probe: fitting in pure numpy, plus JSON persistence.

Per layer l: ``dhat_l = normalize(mean(pos) - mean(neg))``, ``mid_l = midpoint_l . dhat_l`` (the projected
midpoint, a scalar), so the raw projection of any residual x is ``x . dhat_l - mid_l`` with threshold 0.
Sequence z divides a span mean's projection by ``sd_seq_l``; token z divides a token's by ``sd_tok_l``.

A deflated probe also stores an orthonormal basis of confound axes per layer (directions such as warmth, fit
from pairs that hold the concept fixed, one per removal round); every residual is projected onto their
complement, ``x - sum_k (x . n_kl) n_kl``, before any of the above, at fit time and at chat time alike.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import string
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from backend import config
from backend.contract import AxisSet, ContrastSet, Example

VERDICT_CONCEPT = "tracks the concept"
VERDICT_MIXED = "mixed"
VERDICT_WORDS = "tracks the words"
VERDICT_NOT_TESTED = "not tested"  # a stress group is empty, so the index has nothing to compare
CV_FOLDS = 4
PLATEAU_TOLERANCE = 0.01
FIX_BUTTON_DECOYS = 6  # decoys[:6] feed the fix button, so the panel never uses them
STABILITY_SPLITS = 5


# ------------------------------------------------------------------ numpy math


def fit_direction(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pos [Np, L, d], neg [Nn, L, d] -> dhat [L, d] (float16-quantized so stored and live copies agree), mid [L]."""
    mp, mn = pos.mean(axis=0), neg.mean(axis=0)
    d = mp - mn
    dhat = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    dhat = dhat.astype(np.float16).astype(np.float32)
    mid = np.einsum("ld,ld->l", (mp + mn) / 2, dhat)
    return dhat, mid


def project(x: np.ndarray, dhat: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """x [N, L, d] -> raw projections [N, L]."""
    return np.einsum("nld,ld->nl", x, dhat) - mid


def auroc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Rank AUROC per layer from scores pos [Np, L], neg [Nn, L]; ties count half."""
    gt = (pos[:, None, :] > neg[None, :, :]).mean(axis=(0, 1))
    eq = (pos[:, None, :] == neg[None, :, :]).mean(axis=(0, 1))
    return gt + 0.5 * eq


def cv_auroc(pos: np.ndarray, neg: np.ndarray, k: int = CV_FOLDS, seed: int = 0) -> np.ndarray:
    """Mean over k stratified folds of the AUROC of a direction refit on the other folds."""
    rng = np.random.default_rng(seed)
    fold_p = rng.permutation(len(pos)) % k
    fold_n = rng.permutation(len(neg)) % k
    aucs = []
    for f in range(k):
        train_p, train_n = pos[fold_p != f], neg[fold_n != f]
        test_p, test_n = pos[fold_p == f], neg[fold_n == f]
        if min(len(train_p), len(train_n), len(test_p), len(test_n)) == 0:
            continue
        dhat, mid = fit_direction(train_p, train_n)
        aucs.append(auroc(project(test_p, dhat, mid), project(test_n, dhat, mid)))
    if not aucs:
        return np.full(pos.shape[1], 0.5)
    return np.mean(aucs, axis=0)


def accuracy(pos: np.ndarray, neg: np.ndarray, dhat: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Threshold-0 accuracy per layer on pos [Np, L, d] and neg [Nn, L, d]."""
    if len(pos) + len(neg) == 0:
        return np.zeros(dhat.shape[0])
    hits = np.concatenate([project(pos, dhat, mid) > 0, project(neg, dhat, mid) <= 0], axis=0)
    return hits.mean(axis=0)


def heldout_auroc(pos: np.ndarray, neg: np.ndarray, dhat: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """AUROC per layer of a fitted direction on held-out pos [Np, L, d] vs neg [Nn, L, d]; 0.5 when a side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return np.full(dhat.shape[0], 0.5)
    return auroc(project(pos, dhat, mid), project(neg, dhat, mid))


def axis_direction(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Unit axis per layer from paired reps high [P, L, d], low [P, L, d]: the mean within-pair difference,
    float16-quantized like ``dhat`` so the stored and live copies agree."""
    n = (high - low).mean(axis=0)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    return n.astype(np.float16).astype(np.float32)


def deflate(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project every axis of the basis out of every rep: x [N, L, d], basis [k, L, d] of unit rows, one axis
    at a time so a single axis reproduces ``x - (x . axis) axis`` exactly."""
    for axis in basis:
        x = x - np.einsum("nld,ld->nl", x, axis)[..., None] * axis
    return x


RESIDUAL_AXIS_FLOOR = 1e-2  # of the round-one difference norm; float16 quantization alone leaves about 1e-3
MIN_RESIDUAL_DROP = 0.02  # a further removal round must lower the residual separability at the picked layer by this much


def top_principal_direction(rows: np.ndarray) -> np.ndarray:
    """Unit first right singular vector of ``rows`` [P, d], through the P x P Gram matrix since P << d."""
    _, u = np.linalg.eigh(rows @ rows.T)
    v = rows.T @ u[:, -1]
    return v / max(float(np.linalg.norm(v)), 1e-12)


def principal_axis(high: np.ndarray, low: np.ndarray, basis: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """The next axis from pair reps already deflated by ``basis``: per layer the top principal direction of the
    within-pair differences (their mean is zero by construction once the mean axis is out, so no centering),
    with the basis components removed, sign set so high minus low projects positive on average, unit and
    float16-quantized like the basis. A layer whose remaining differences are under ``RESIDUAL_AXIS_FLOOR``
    times ``reference`` (that layer's round-one Frobenius norm) is exhausted and gets a zero row, which
    deflates nothing; None when every layer is exhausted."""
    d = high - low
    for earlier in basis:
        d = d - np.einsum("pld,ld->pl", d, earlier)[..., None] * earlier
    alive = np.linalg.norm(d, axis=(0, 2)) > RESIDUAL_AXIS_FLOOR * reference
    if not alive.any():
        return None
    axes = np.zeros(d.shape[1:], dtype=np.float32)
    for layer in np.flatnonzero(alive):
        v = top_principal_direction(d[:, layer, :])
        for earlier in basis:
            v = v - (v @ earlier[layer]) * earlier[layer]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        if (d[:, layer, :] @ v).mean() < 0:
            v = -v
        axes[layer] = v
    return axes.astype(np.float16).astype(np.float32)


def axis_cv_auroc(high: np.ndarray, low: np.ndarray, deflated: bool, estimator: str = "mean") -> np.ndarray:
    """Leave-two-out CV AUROC of high vs low per layer from paired reps high [P, L, d], low [P, L, d]: for every
    two pairs, one pair's difference is the direction and the other pair is read along it (1 when its high
    lands above its low, 0.5 on a tie, 0 otherwise), averaged over all pairs of pairs. With ``deflated`` both
    differences are first deflated by an axis fit on the remaining pairs, so neither the direction nor the
    pair being read touched the axis: a direction fit on the pairs that fit the axis is zero by
    construction, and one fit on the complement of an axis that saw the read pair is anti-aligned with it.
    Near 0.5 after deflation means one direction carried the axis; higher means the pairs still agree on a
    residual direction. ``estimator`` names how the held-out axis is fit: the mean difference of the remaining
    pairs, or their top principal direction (for reps whose mean difference is already out)."""
    d = high - low
    n_pairs = len(d)
    if n_pairs < (3 if deflated else 2):
        return np.full(d.shape[1], 0.5)
    total = d.sum(axis=0)
    scores = []
    for f in range(n_pairs):
        for g in range(f + 1, n_pairs):
            df, dg = d[f], d[g]
            if deflated:
                if estimator == "mean":
                    n = total - df - dg
                    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
                else:
                    others = np.delete(d, [f, g], axis=0)
                    n = np.stack([top_principal_direction(others[:, layer, :]) for layer in range(d.shape[1])])
                df = df - np.einsum("ld,ld->l", df, n)[:, None] * n
                dg = dg - np.einsum("ld,ld->l", dg, n)[:, None] * n
            agree = np.einsum("ld,ld->l", df, dg)
            scores.append((agree > 0) + 0.5 * (agree == 0))
    return np.mean(scores, axis=0)


def _halves(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(x))
    return x[order[: len(x) // 2]], x[order[len(x) // 2 :]]


def stability(pos: np.ndarray, matched: np.ndarray, background: np.ndarray, neutral: np.ndarray, n_splits: int = STABILITY_SPLITS) -> dict | None:
    """Split-half stability of the shipped direction per layer, from pooled reps [N, L, d]: is the set big enough?

    Each training pool is halved at random ``n_splits`` times; a direction is fit on each half with the shipped
    negatives rule (its matched + background + neutral halves). ``split_half_cosine`` is the mean cosine between
    the two half-directions, ``full_set_reliability`` its Spearman-Brown projection 2r / (1 + r) to the full
    set, and ``split_half_auroc`` reads one half's direction on the other half's positives vs matched negatives,
    both orders. None when a pool is too small to halve."""
    cosines, aucs = [], []
    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        halves = [_halves(x, rng) for x in (pos, matched, background, neutral)]
        sides = []
        for side in (0, 1):
            p, m, b, n = (h[side] for h in halves)
            neg = np.concatenate([m, b, n])
            if len(p) == 0 or len(m) == 0:
                break
            sides.append((p, m, *fit_direction(p, neg)))
        if len(sides) < 2:
            continue
        (pos_a, matched_a, dhat_a, mid_a), (pos_b, matched_b, dhat_b, mid_b) = sides
        norms = np.linalg.norm(dhat_a, axis=-1) * np.linalg.norm(dhat_b, axis=-1)
        cosines.append(np.einsum("ld,ld->l", dhat_a, dhat_b) / np.maximum(norms, 1e-12))
        aucs.append((heldout_auroc(pos_b, matched_b, dhat_a, mid_a) + heldout_auroc(pos_a, matched_a, dhat_b, mid_b)) / 2)
    if not cosines:
        return None
    r = np.mean(cosines, axis=0)
    return {
        "split_half_cosine": _round(r),
        "full_set_reliability": _round(2 * r / np.maximum(1 + r, 1e-6)),
        "split_half_auroc": _round(np.mean(aucs, axis=0)),
        "n_splits": len(cosines),
    }


def plateau_onset(curve: np.ndarray, tolerance: float = PLATEAU_TOLERANCE) -> int:
    """First layer within ``tolerance`` of the curve's maximum (never argmax, which flips on a flat plateau).

    The search starts a quarter of the way into the stack: a lexical concept can saturate the
    curve in the first few layers, and a direction read there is a word detector that steers
    the model into collapse. The slider still reaches every layer."""
    floor = len(curve) // 4
    tail = curve[floor:]
    return floor + int(np.argmax(tail >= tail.max() - tolerance))


def verdict(index: float) -> str:
    if index > 1.0:
        return VERDICT_CONCEPT
    if index >= 0.3:
        return VERDICT_MIXED
    return VERDICT_WORDS


def group_mean(seq_z: np.ndarray, n_layers: int) -> list[float]:
    return _round(seq_z.mean(axis=0)) if len(seq_z) else [0.0] * n_layers


def _round(a: np.ndarray, digits: int = 3) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


# ------------------------------------------------------------------ contrast-set helpers


def has_keyword(text: str, keywords: list[str]) -> bool:
    """Case-insensitive stem match anchored at a word start, the rule the generator and the frontend share:
    "sad" hits "sadness", "wise" does not hit "otherwise"."""
    low = text.lower()
    return any(re.search(r"\b" + re.escape(k.strip().lower()), low) for k in keywords if k.strip())


def panel_decoys(cs: ContrastSet) -> list[Example]:
    """Decoys the panel measures: never the fix button's six, and never one that is already a training negative."""
    candidates = cs.decoys[FIX_BUTTON_DECOYS:] if len(cs.decoys) >= 2 * FIX_BUTTON_DECOYS else list(cs.decoys)
    training = {e.text.strip() for e in cs.train_neg}
    return [e for e in candidates if e.text.strip() not in training]


def fix_button_decoys(cs: ContrastSet) -> list[Example]:
    """The decoys the fix button adds as negatives, ``decoys[:6]`` minus any text already in train_neg; never the
    held-out panel's. Empty when the set is too small for the panel to have its own six."""
    if len(cs.decoys) < 2 * FIX_BUTTON_DECOYS:
        return []
    training = {e.text.strip() for e in cs.train_neg}
    return [e for e in cs.decoys[:FIX_BUTTON_DECOYS] if e.text.strip() not in training]


def probe_id_for(model_id: str, cs: ContrastSet, axis_set: AxisSet | None = None, rounds: int = 1) -> str:
    """Stable id from the model, the contrast set, the axis set, and the number of removal rounds. A single
    round leaves the payload as it was, so those ids do not change."""
    payload = {"model": model_id, "set": cs.model_dump()}
    if axis_set is not None:
        payload["axis"] = axis_set.model_dump()
        if rounds > 1:
            payload["rounds"] = rounds
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def _encode_f16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a.astype(np.float16)).tobytes()).decode("ascii")


def _decode_f16(b64: str, *shape: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float16).reshape(*shape).astype(np.float32)


# ------------------------------------------------------------------ the probe object


@dataclass
class Probe:
    probe_id: str
    model_id: str
    n_layers: int
    picked_layer: int
    dhat: np.ndarray  # [L, d] float32
    mid: np.ndarray  # [L]
    sd_seq: np.ndarray  # [L]
    sd_tok: np.ndarray  # [L]
    mean_norm: np.ndarray  # [L]
    fit: dict  # FitResponse fields
    contrast_set: dict = field(default_factory=dict)
    basis: np.ndarray | None = None  # [k, L, d] orthonormal confound axes projected out of every rep before scoring

    def seq_z(self, proj_mean: np.ndarray) -> np.ndarray:
        return proj_mean / self.sd_seq

    def tok_z(self, proj: np.ndarray) -> np.ndarray:
        return proj / self.sd_tok

    def to_json(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "model_id": self.model_id,
            "n_layers": self.n_layers,
            "d_model": int(self.dhat.shape[1]),
            "picked_layer": self.picked_layer,
            "dhat_f16_b64": _encode_f16(self.dhat),
            "axis_basis_f16_b64": _encode_f16(self.basis) if self.basis is not None else None,
            "n_axes": int(self.basis.shape[0]) if self.basis is not None else 0,
            "mid": self.mid.tolist(),
            "sd_seq": self.sd_seq.tolist(),
            "sd_tok": self.sd_tok.tolist(),
            "mean_norm": self.mean_norm.tolist(),
            "fit": self.fit,
            "contrast_set": self.contrast_set,
        }

    @classmethod
    def from_json(cls, j: dict) -> Probe:
        L, d = j["n_layers"], j["d_model"]
        if j.get("axis_basis_f16_b64"):
            basis = _decode_f16(j["axis_basis_f16_b64"], j["n_axes"], L, d)
        elif j.get("axis_f16_b64"):  # files written with a single axis
            basis = _decode_f16(j["axis_f16_b64"], 1, L, d)
        else:
            basis = None
        return cls(
            probe_id=j["probe_id"],
            model_id=j["model_id"],
            n_layers=j["n_layers"],
            picked_layer=j["picked_layer"],
            dhat=_decode_f16(j["dhat_f16_b64"], j["n_layers"], j["d_model"]),
            mid=np.asarray(j["mid"], dtype=np.float32),
            sd_seq=np.asarray(j["sd_seq"], dtype=np.float32),
            sd_tok=np.asarray(j["sd_tok"], dtype=np.float32),
            mean_norm=np.asarray(j["mean_norm"], dtype=np.float32),
            fit=j["fit"],
            contrast_set=j["contrast_set"],
            basis=basis,
        )


class ProbeStore:
    """Probes as JSON files under ``{DATA_DIR}/probes``, with an in-memory cache."""

    def __init__(self, data_dir: str = config.DATA_DIR) -> None:
        self.dir = os.path.join(data_dir, "probes")
        os.makedirs(self.dir, exist_ok=True)
        self._cache: dict[str, Probe] = {}

    def path(self, probe_id: str) -> str:
        return os.path.join(self.dir, f"{probe_id}.json")

    def save(self, probe: Probe) -> None:
        tmp = self.path(probe.probe_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(probe.to_json(), f, ensure_ascii=False)
        os.replace(tmp, self.path(probe.probe_id))
        self._cache[probe.probe_id] = probe

    def get(self, probe_id: str) -> Probe | None:
        if probe_id in self._cache:
            return self._cache[probe_id]
        safe = probe_id.replace("/", "").replace("..", "")
        if not safe or not os.path.exists(self.path(safe)):
            return None
        with open(self.path(safe), encoding="utf-8") as f:
            probe = Probe.from_json(json.load(f))
        self._cache[probe_id] = probe
        return probe


# ------------------------------------------------------------------ fit pipeline


@dataclass
class EncodedSet:
    """One forward pass over a contrast set: pooled reps [N, L, d], per-token GPU reps and rendered prompts
    for the training examples, and the slice of ``pooled`` each list occupies."""

    pooled: np.ndarray
    tokens: list
    rendered: list
    offsets: dict[str, slice]
    decoys: list[Example]


def encode_contrast_set(runner, cs: ContrastSet) -> EncodedSet:
    """Encode every example once. Training lists come first so their token reps can be kept for
    ``sd_tok`` and ``mean_norm``."""
    decoys = panel_decoys(cs)
    lists = [
        ("train_pos", cs.train_pos),
        ("train_neg", cs.train_neg),
        ("background", cs.background),
        ("neutral", cs.neutral),
        ("heldout_pos", cs.heldout_pos),
        ("heldout_neg", cs.heldout_neg),
        ("implicit_pos", cs.implicit_pos),
        ("decoys", decoys),
    ]
    examples = [e for _, lst in lists for e in lst]
    rendered = [runner.render_example(e.text, e.context) for e in examples]
    n_train = len(cs.train_pos) + len(cs.train_neg) + len(cs.background) + len(cs.neutral)
    pooled, tokens = runner.encode_examples(rendered, keep_tokens=n_train)
    offsets, at = {}, 0
    for name, lst in lists:
        offsets[name] = slice(at, at + len(lst))
        at += len(lst)
    return EncodedSet(pooled, tokens[:n_train], rendered[:n_train], offsets, decoys)


def fit_from_reps(runner, concept: str, cs: ContrastSet, enc: EncodedSet, axis_set: AxisSet | None = None, basis: np.ndarray | None = None) -> Probe:
    """Fit the shipped direction (positives vs matched negatives + background + neutral), evaluate it and the
    matched-only and background-only candidates on the same matched held-out set from the same reps, then
    score the stress groups. With ``basis`` ([k, L, d] unit rows) every rep, pooled and per token, is
    deflated first."""
    pooled = deflate(enc.pooled, basis) if basis is not None else enc.pooled
    offsets = enc.offsets
    pos = pooled[offsets["train_pos"]]
    matched = pooled[offsets["train_neg"]]
    background = np.concatenate([pooled[offsets["background"]], pooled[offsets["neutral"]]])
    neg = np.concatenate([matched, background])
    dhat, mid = fit_direction(pos, neg)
    train_proj = project(np.concatenate([pos, neg]), dhat, mid)
    sd_seq = np.maximum(train_proj.std(axis=0), 1e-6)
    tok_proj, tok_norm = runner.project_tokens(enc.tokens, dhat, mid, basis)
    sd_tok = np.maximum(tok_proj.std(axis=0), 1e-6)
    mean_norm = tok_norm.mean(axis=0)

    curve = cv_auroc(pos, neg)
    picked = plateau_onset(curve)
    heldout_pos, heldout_neg = pooled[offsets["heldout_pos"]], pooled[offsets["heldout_neg"]]
    heldout_acc = accuracy(heldout_pos, heldout_neg, dhat, mid)
    heldout = {"shipped": heldout_auroc(heldout_pos, heldout_neg, dhat, mid)}
    for name, candidate_neg in (("matched", matched), ("background", background)):
        if len(candidate_neg) == 0:
            heldout[name] = np.full(runner.n_layers, 0.5)
        else:
            heldout[name] = heldout_auroc(heldout_pos, heldout_neg, *fit_direction(pos, candidate_neg))

    seq_z_all = project(pooled, dhat, mid) / sd_seq
    train_pos_z = seq_z_all[offsets["train_pos"]]
    explicit_mask = np.array([has_keyword(e.text, cs.keywords) for e in cs.train_pos], dtype=bool)
    group_z = {
        "explicit": train_pos_z[explicit_mask] if len(cs.train_pos) else train_pos_z,
        "implicit": seq_z_all[offsets["implicit_pos"]],
        "decoys": seq_z_all[offsets["decoys"]],
        "neutral": seq_z_all[offsets["neutral"]],
    }
    means = {g: group_mean(z, runner.n_layers) for g, z in group_z.items()}
    tested = len(cs.implicit_pos) > 0 and len(enc.decoys) > 0
    index = np.asarray(means["implicit"]) - np.asarray(means["decoys"]) if tested else np.zeros(runner.n_layers)
    verdicts = [verdict(float(v)) for v in index] if tested else [VERDICT_NOT_TESTED] * runner.n_layers

    explicit = [e for e in cs.train_pos if has_keyword(e.text, cs.keywords)]
    groups = [("explicit", explicit), ("implicit", cs.implicit_pos), ("decoys", enc.decoys), ("neutral", cs.neutral)]
    stress = []
    for name, lst in groups:
        for e, z in zip(lst, group_z[name]):
            stress.append({"text": e.text, "context": e.context, "group": name, "seq_z": _round(z)})

    probe_id = probe_id_for(config.MODEL_ID, cs, axis_set, len(basis) if basis is not None else 1)
    fit = {
        "probe_id": probe_id,
        "concept": concept,
        "model_id": config.MODEL_ID,
        "n_layers": runner.n_layers,
        "picked_layer": picked,
        "per_layer": {
            "cv_auroc": _round(curve),
            "heldout_acc": _round(heldout_acc),
            "heldout_auroc": _round(heldout["shipped"]),
            "heldout_auroc_matched": _round(heldout["matched"]),
            "heldout_auroc_background": _round(heldout["background"]),
        },
        "confound": {"index": _round(index), "means": means},
        "verdict_by_layer": verdicts,
        "stress_examples": stress,
        "deflated_axis": axis_set.axis if axis_set is not None else None,
        "stability": stability(pos, matched, pooled[offsets["background"]], pooled[offsets["neutral"]]),
    }
    return Probe(
        probe_id=probe_id,
        model_id=config.MODEL_ID,
        n_layers=runner.n_layers,
        picked_layer=picked,
        dhat=dhat,
        mid=mid.astype(np.float32),
        sd_seq=sd_seq.astype(np.float32),
        sd_tok=sd_tok.astype(np.float32),
        mean_norm=mean_norm.astype(np.float32),
        fit=fit,
        contrast_set=cs.model_dump(),
        basis=basis,
    )


def fit_probe(runner, concept: str, cs: ContrastSet) -> Probe:
    return fit_from_reps(runner, concept, cs, encode_contrast_set(runner, cs))


# ------------------------------------------------------------------ misfires


MISFIRE_SEEDS = 12
MISFIRE_SEED_POOL = 3  # seeds come from the top ``MISFIRE_SEED_POOL * top_k`` tokens
MISFIRE_SEED_MIN_LETTERS = 3
MISFIRE_STOPWORDS = frozenset(
    """a an the i me my mine you your yours he him his she her hers it its we us our ours they them their theirs
    this that these those who whom whose which what am is are was were be been being have has had having do does
    did doing will would shall should can could may might must ought need of in on at to for with from by about as
    into like through after over between out against during without before under around among up down off than
    then and but or nor so yet because if while although though whether either neither both yes no not just also
    very really there here when where why how all any each few more most other some such only own same too
    isn aren wasn weren don doesn didn hasn haven hadn won wouldn couldn shouldn ll re ve""".split()
)


def misfires(runner, probe: Probe, cs: ContrastSet, layer: int, top_k: int) -> dict:
    """The highest-z tokens among the concept-absent examples at ``layer``: what the direction fires on when the
    concept is absent. Groups: matched negatives, background, neutral, and the fix button's decoys (which carry
    the confound vehicle by construction; the held-out panel decoys are never read). Token z uses the probe's
    mid, sd_tok and stored axis on the same encoding path as the fit. The first token of every span is skipped
    because the opening word of a turn is a position effect. ``seeds`` are distinct cleaned token strings from
    the top ``MISFIRE_SEED_POOL * top_k`` tokens, function words dropped, ordered by how many distinct examples
    they fire in, for the decoy generator."""
    enc = encode_contrast_set(runner, cs)
    items = [(group, enc.rendered[i], enc.tokens[i]) for group, name in (("matched", "train_neg"), ("background", "background"), ("neutral", "neutral")) for i in range(enc.offsets[name].start, enc.offsets[name].stop)]
    fix = fix_button_decoys(cs)
    if fix:
        rendered = [runner.render_example(e.text, e.context) for e in fix]
        _, tokens = runner.encode_examples(rendered, keep_tokens=len(rendered))
        items += [("decoys", r, t) for r, t in zip(rendered, tokens)]
    if not items:
        return {"layer": layer, "misfires": [], "seeds": []}
    proj, _ = runner.project_tokens([reps for _, _, reps in items], probe.dhat, probe.mid, probe.basis)
    z_all = proj[:, layer] / probe.sd_tok[layer]

    scored, at = [], 0
    for example, (group, rendered, _) in enumerate(items):
        span = rendered.spans[0]
        ids = runner.encode_ids(rendered.prompt)[span.start : span.end]
        z = z_all[at : at + len(ids)]
        at += len(ids)
        for token_id, value in list(zip(ids, z))[1:]:
            token = runner.tok.decode([token_id])
            if any(c.isalnum() for c in token):
                scored.append({"token": token, "z": float(value), "text": span.text, "group": group, "example": example})
    scored.sort(key=lambda m: -m["z"])

    examples_by_seed: dict[str, set[int]] = {}
    best_z: dict[str, float] = {}
    for m in scored[: MISFIRE_SEED_POOL * top_k]:
        clean = m["token"].strip().lower().strip(string.punctuation)
        if clean in MISFIRE_STOPWORDS or sum(c.isalpha() for c in clean) < MISFIRE_SEED_MIN_LETTERS:
            continue
        examples_by_seed.setdefault(clean, set()).add(m["example"])
        best_z[clean] = max(best_z.get(clean, m["z"]), m["z"])
    seeds = sorted(examples_by_seed, key=lambda t: (-len(examples_by_seed[t]), -best_z[t], t))[:MISFIRE_SEEDS]
    top = [{"token": m["token"], "z": round(m["z"], 3), "text": m["text"], "group": m["group"]} for m in scored[:top_k]]
    return {"layer": layer, "misfires": top, "seeds": seeds}


# ------------------------------------------------------------------ deflation


@dataclass
class Deflation:
    """A probe refit with a confound axis projected out, plus the evidence per layer."""

    probe: Probe
    axis_cv_auroc_before: list[float]
    axis_cv_auroc_after: list[float]
    heldout_auroc_before: list[float]
    heldout_auroc_after: list[float]
    confound_index_before: list[float]
    confound_index_after: list[float]
    rounds_applied: int
    round_history: list[dict]  # DeflateRound fields: the state after each round


def _describe_drop(drop: float, layer: int) -> str:
    if drop < 0:
        return f"raise the residual separability at layer {layer} by {-drop:.3f}"
    return f"lower the residual separability at layer {layer} by only {drop:.3f}"


def deflate_probe(runner, concept: str, cs: ContrastSet, axis_set: AxisSet, rounds: int = 1) -> Deflation:
    """Fit the axis as the mean over every pair of pooled(high) minus pooled(low): the concept is held fixed
    within each pair, so it cancels and the axis remains. Then refit the concept on reps with that axis
    projected out. Round one is the mean difference; each further round takes the top principal direction of
    the pair differences deflated by the axes so far (``principal_axis``), since their mean is zero by then.
    A further round is applied only if its leave-two-out residual separability at the stored probe's picked
    layer is at least ``MIN_RESIDUAL_DROP`` below the previous round's; otherwise, or once every layer's
    residual is exhausted, the loop stops and ``rounds_applied`` says so. The history records the residual
    separability, the concept held-out AUROC and the confound index after every applied round, with a note on
    the estimator and on why the loop stopped. The contrast set is encoded in its own pass, the same batches
    as ``fit_probe``, so the "before" fit reproduces the stored probe and both fits read the same reps."""
    enc = encode_contrast_set(runner, cs)
    pairs = axis_set.absent_pairs + axis_set.present_pairs
    rendered = [runner.render_example(p.high, p.context) for p in pairs] + [runner.render_example(p.low, p.context) for p in pairs]
    pooled, _ = runner.encode_examples(rendered)
    high, low = pooled[: len(pairs)], pooled[len(pairs) :]

    before = fit_from_reps(runner, concept, cs, enc)
    pick = before.picked_layer
    reference = np.linalg.norm(high - low, axis=(0, 2))
    previous = axis_cv_auroc(high, low, deflated=False)
    basis = np.zeros((0,) + high.shape[1:], dtype=np.float32)
    history = []
    for round_number in range(1, rounds + 1):
        high_r, low_r = deflate(high, basis), deflate(low, basis)
        if round_number == 1:
            axis, estimator, note = axis_direction(high_r, low_r), "mean", "axis: mean difference of the pairs"
        else:
            axis, estimator, note = principal_axis(high_r, low_r, basis, reference), "principal", "axis: top principal direction of the deflated differences"
            if axis is None:
                history[-1]["note"] += "; stopped: every layer's residual is exhausted"
                break
        residual = axis_cv_auroc(high_r, low_r, deflated=True, estimator=estimator)
        drop = float(previous[pick] - residual[pick])
        if round_number > 1 and drop < MIN_RESIDUAL_DROP:
            history[-1]["note"] += f"; stopped: another round would {_describe_drop(drop, pick)}"
            break
        basis = np.concatenate([basis, axis[None]])
        after = fit_from_reps(runner, concept, cs, enc, axis_set, basis)
        history.append({
            "round": round_number,
            "axis_cv_auroc": _round(residual),
            "heldout_auroc": after.fit["per_layer"]["heldout_auroc"],
            "confound_index": after.fit["confound"]["index"],
            "note": note,
        })
        previous = residual
        if round_number < rounds and drop < MIN_RESIDUAL_DROP:
            history[-1]["note"] += f"; stopped: this round did no better than {_describe_drop(drop, pick)}"
            break
    return Deflation(
        probe=after,
        axis_cv_auroc_before=_round(axis_cv_auroc(high, low, deflated=False)),
        axis_cv_auroc_after=history[-1]["axis_cv_auroc"],
        heldout_auroc_before=before.fit["per_layer"]["heldout_auroc"],
        heldout_auroc_after=after.fit["per_layer"]["heldout_auroc"],
        confound_index_before=before.fit["confound"]["index"],
        confound_index_after=after.fit["confound"]["index"],
        rounds_applied=len(basis),
        round_history=history,
    )
