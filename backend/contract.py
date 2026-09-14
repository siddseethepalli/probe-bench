"""Wire contract shared by every route. frontend/src/types.ts mirrors these shapes exactly."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Group = Literal["explicit", "implicit", "decoys", "neutral"]
Role = Literal["system", "user", "assistant"]


class Example(BaseModel):
    text: str
    context: str | None = None  # user turn the text replies to; required for response-property concepts


class ContrastSet(BaseModel):
    concept: str
    is_response_property: bool
    keywords: list[str]
    train_pos: list[Example]  # 32: 16 explicit (use a keyword), 16 implicit (no keyword)
    train_neg: list[Example]  # 32, matched one for one in topic and length, not tone
    heldout_pos: list[Example]
    heldout_neg: list[Example]
    implicit_pos: list[Example]
    decoys: list[Example]  # first 6 feed the fix button, last 6 stay held out for the panel
    neutral: list[Example]
    confound_axis: str | None = None  # the tone or style axis the stress sets decouple, e.g. "warmth"; named by the generator
    background: list[Example]  # 24: same style and length as the positives, unrelated everyday topics, no keywords, no single alternative concept; replies to varied prompts for response-property concepts
    system_prompts: list[str]  # [raise, lower]
    demo_user_message: str


class ConceptRequest(BaseModel):
    concept: str


class ConceptResponse(BaseModel):
    contrast_set: ContrastSet
    dropped: dict[str, int]  # per list, shortfall against the contract size after validation (0 means the list is full)
    cached: bool


# /concept streams server-sent events, because generation can take longer than the proxy's
# 100 s first-byte limit. Each `data:` line is one JSON object:
#   {"type": "status", "text": "..."}        heartbeat with a human-readable stage, every 10 s
#   {"type": "concept", ...ConceptResponse}   once, last
#   {"type": "error", "error": "..."}         instead of concept


class FitRequest(BaseModel):
    concept: str
    contrast_set: ContrastSet


class PerLayer(BaseModel):
    cv_auroc: list[float]  # [L] shipped direction (positives vs matched + background + neutral), 4-fold CV on its own pool
    heldout_acc: list[float]  # [L] shipped direction, threshold-0 accuracy on the matched held-out set
    heldout_auroc: list[float]  # [L] shipped direction, AUROC on the matched held-out set (heldout_pos vs heldout_neg)
    heldout_auroc_matched: list[float]  # [L] direction fit on matched negatives only, same held-out set
    heldout_auroc_background: list[float]  # [L] direction fit on background + neutral only, same held-out set


class GroupMeans(BaseModel):
    explicit: list[float]  # [L] mean seq_z over the explicit positives (train_pos that use keywords)
    implicit: list[float]
    decoys: list[float]  # panel decoys only (the held-out six)
    neutral: list[float]  # in-sample: neutral examples are training negatives; shown as a reference bar, not a test


class Confound(BaseModel):
    index: list[float]  # [L] implicit minus decoys
    means: GroupMeans


class StressExample(BaseModel):
    text: str
    context: str | None = None
    group: Group
    seq_z: list[float]  # [L]


class Stability(BaseModel):
    """Is the contrast set big enough? Random halves of the training pools are fit separately."""

    split_half_cosine: list[float]  # [L] mean cosine between the two half-directions over the splits
    full_set_reliability: list[float]  # [L] Spearman-Brown projection of the cosine to the full set: 2r / (1 + r)
    split_half_auroc: list[float]  # [L] direction from one half, AUROC on the other half's positives vs negatives
    n_splits: int


class FitResponse(BaseModel):
    probe_id: str
    concept: str
    model_id: str
    n_layers: int
    picked_layer: int  # index into [0, n_layers)
    per_layer: PerLayer
    confound: Confound
    verdict_by_layer: list[str]  # "tracks the concept" | "mixed" | "tracks the words" | "not tested" (a stress group is empty)
    stress_examples: list[StressExample]
    deflated_axis: str | None = None  # set when the direction had a confound axis projected out
    stability: Stability | None = None  # computed from the pooled reps at fit time, no extra GPU work


class ProbeInfo(FitResponse):
    contrast_set: ContrastSet


class AxisPair(BaseModel):
    """Two texts with the same content and the confound axis flipped: warm and cold, formal and casual."""

    high: str  # axis present (warm)
    low: str  # axis absent (cold)
    context: str | None = None


class AxisSet(BaseModel):
    """Pairs for fitting a confound-axis direction that cancels the concept: half the pairs have the
    concept absent, half present, so the mean of (high minus low) carries the axis and not the concept."""

    axis: str
    absent_pairs: list[AxisPair]  # 12, concept absent in both texts
    present_pairs: list[AxisPair]  # 12, concept present in both texts


class AxisRequest(BaseModel):
    concept: str
    contrast_set: ContrastSet


class AxisResponse(BaseModel):
    axis_set: AxisSet
    dropped: dict[str, int]
    cached: bool


# /axis streams server-sent events like /concept: {"type":"status"} heartbeats, then
# {"type":"axis", ...AxisResponse}, or {"type":"error"}.


class DeflateRequest(BaseModel):
    probe_id: str
    axis_set: AxisSet
    rounds: int = 1  # 1 to 4. Round 1 removes the mean difference of the pairs; the mean of the deflated differences is then zero by construction, so later rounds remove the top principal direction of what remains, stopping when the residual separability stops dropping


class DeflateRound(BaseModel):
    round: int
    axis_cv_auroc: list[float]  # [L] residual separability after this round
    heldout_auroc: list[float]  # [L] concept held-out AUROC after this round
    confound_index: list[float]  # [L]
    note: str | None = None  # which estimator produced the round's axis, and why the loop stopped when it did


class DeflateResponse(BaseModel):
    """The probe refit on activations with the axis direction projected out, plus the evidence."""

    fit: FitResponse  # a new probe (new probe_id, deflated_axis set) usable by /chat and /score
    axis: str
    axis_cv_auroc_before: list[float]  # [L] leave-two-out high-vs-low separability: axis fit on the other pairs, one held-out pair gives the direction, the other is read along it
    axis_cv_auroc_after: list[float]  # [L] the same on deflated reps; near 0.5 means the axis is gone, higher means a second axis direction remains
    heldout_auroc_before: list[float]  # [L] concept held-out AUROC of the original direction
    heldout_auroc_after: list[float]  # [L] of the deflated direction
    confound_index_before: list[float]  # [L]
    confound_index_after: list[float]  # [L]
    rounds_applied: int = 1
    round_history: list[DeflateRound] = Field(default_factory=list)  # state after each round, so the cost curve is visible


class MisfiresRequest(BaseModel):
    probe_id: str
    layer: int | None = None  # defaults to the probe's picked layer
    top_k: int = 20


class Misfire(BaseModel):
    token: str
    z: float
    text: str  # the concept-absent training example the token came from
    group: Literal["matched", "background", "neutral", "decoys"]  # decoys = the first six (fix-button) decoys, never the held-out panel six


class MisfiresResponse(BaseModel):
    """The highest-scoring tokens among concept-absent training examples: what the direction fires on
    when the concept is absent, which is the vehicle a naive probe rides."""

    layer: int
    misfires: list[Misfire]
    seeds: list[str]  # distinct cleaned token strings for the decoy generator, most frequent first


class MineRequest(BaseModel):
    concept: str
    contrast_set: ContrastSet
    seeds: list[str]


class MineResponse(BaseModel):
    decoys: list[Example]  # 12 concept-absent examples built around the seed tokens, same format rules as the set
    dropped: dict[str, int]
    cached: bool


# /mine streams server-sent events like /axis: {"type":"status"} heartbeats, then
# {"type":"mined", ...MineResponse}, or {"type":"error"}. The client appends the mined decoys to
# train_neg tagged "mined" and refits; the held-out panel decoys stay, so the index is comparable.


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    probe_id: str
    system_prompt: str = ""
    messages: list[Message]  # history plus the new user message last
    alpha: float = 0.0  # steering strength in units of the layer's mean residual norm; 0 = off
    layer: int | None = None  # defaults to the probe's picked layer
    max_new_tokens: int = 400
    temperature: float = 0.7
    seed: int = 0


class ScoredTurn(BaseModel):
    role: Role
    text: str
    tokens: list[str]  # display strings, one per token
    z: list[list[float]]  # [T][L]
    seq_z: list[float]  # [L]


class ChatScored(BaseModel):
    """Final SSE event of /chat. Every turn of the conversation is returned scored, system prompt
    first when non-empty, so an edited system prompt never leaves stale colors behind."""

    reply: str
    turns: list[ScoredTurn]
    hit_length_cap: bool
    conversation_tokens: int
    flags: list[str]  # e.g. "degenerate", "template_leak"


# /chat streams server-sent events, each `data:` line one JSON object:
#   {"type": "token", "text": "..."}       repeated
#   {"type": "scored", ...ChatScored}       once, last
#   {"type": "error", "error": "..."}       instead of scored


class ScoreRequest(BaseModel):
    probe_id: str
    system_prompt: str = ""
    messages: list[Message]


class ScoreResponse(BaseModel):
    turns: list[ScoredTurn]
    conversation_tokens: int


# Access: every route except GET /health and GET /auth requires the header X-Probe-Key equal to the
# DEMO_PASSWORD environment variable (constant-time compare); otherwise 401 {"error": "password required"}.
# GET /auth answers 200 {"ok": true} with a valid header and 401 without, so the client can validate a
# password before storing it. When DEMO_PASSWORD is unset the server is open (local development).


class Health(BaseModel):
    model_loaded: bool
    loading: bool
    model_id: str
    n_layers: int


class ErrorResponse(BaseModel):
    error: str


class StaticConversation(BaseModel):
    label: str
    system_prompt: str
    user_message: str
    scored: ChatScored


class StaticExample(BaseModel):
    """Shape of frontend/public/examples/<slug>.json, written by scripts/precompute.py."""

    slug: str
    title: str
    blurb: str
    contrast_set: ContrastSet
    fit: FitResponse
    conversations: list[StaticConversation] = Field(default_factory=list)
