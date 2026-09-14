"""Contrast-set generation: a concept string in, a validated ContrastSet out, via Claude structured output.

Two stages. Stage 1 is a small plan call (is_response_property, keywords, topics, tone axis, system prompts,
demo message). Stage 2 runs three concurrent calls that each write part of the set from the same plan: train
lists, stress lists, background pool. The parts are merged and post-validated (keyword, context, length and
duplicate rules); a stage-2 call whose lists fell short is retried once. Results are cached on disk and counted
against a daily cap. Run `python -m backend.generate "<concept>" --out path` to use it from a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pydantic
from dotenv import load_dotenv
from pydantic import BaseModel

from backend.config import (
    CONCEPT_MAX_CHARS,
    DAILY_GENERATION_CAP,
    DATA_DIR,
    GEN_EFFORT,
    GEN_MAX_TOKENS,
    GEN_MODEL,
    GEN_PLAN_EFFORT,
    GEN_PLAN_MODEL,
    GEN_TIMEOUT_S,
)
from backend.contract import AxisPair, AxisResponse, AxisSet, ConceptResponse, ContrastSet, Example, MineResponse

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SLOW_GENERATION_S = 60

LIST_TARGETS = {  # how many the prompt asks for; the stress lists are over-requested so the panel stays full after drops
    "train_pos": 32,
    "train_neg": 32,
    "heldout_pos": 6,
    "heldout_neg": 6,
    "implicit_pos": 8,
    "decoys": 16,
    "neutral": 6,
    "background": 24,
}
LIST_SIZES = {**LIST_TARGETS, "implicit_pos": 6, "decoys": 12}  # what the contract expects; survivors are truncated to this
LIST_MINIMUMS = {
    "train_pos": 24,
    "train_neg": 24,
    "heldout_pos": 4,
    "heldout_neg": 4,
    "implicit_pos": 4,
    "decoys": 4,
    "neutral": 4,
    "background": 16,
}
TRAIN_HALF_SIZE = 16  # each train call writes one half: explicit positives or implicit positives, with matched negatives
TRAIN_HALF_MINIMUM = 12
MIN_WORDS = 4
MAX_WORDS = 60
MIN_KEYWORD_CHARS = 3
AXIS_LIST_TARGETS = {"absent_pairs": 12, "present_pairs": 12}
MINE_TARGET = 16  # over-requested, truncated to MINE_SIZE after validation
MINE_SIZE = 12
MINE_MINIMUM = 4
AXIS_LIST_MINIMUM = 8
AXIS_MAX_LENGTH_GAP = 5  # words between high and low


class GenerationPlan(BaseModel):
    is_response_property: bool
    keywords: list[str]
    tone_axis: str
    confound_axis: str | None
    train_topics: list[str]
    heldout_topics: list[str]
    stress_topics: list[str]
    system_prompts: list[str]
    demo_user_message: str


class TrainLists(BaseModel):
    train_pos: list[Example]
    train_neg: list[Example]


class StressLists(BaseModel):
    heldout_pos: list[Example]
    heldout_neg: list[Example]
    implicit_pos: list[Example]
    decoys: list[Example]
    neutral: list[Example]


class BackgroundList(BaseModel):
    background: list[Example]


class MinedDecoys(BaseModel):
    decoys: list[Example]


SYSTEM_PROMPT = """You write contrast sets for training linear probes on a language model's hidden states. A contrast set is a small dataset of short texts in which a concept is present versus absent. It is used twice: to fit a direction that detects the concept, and to test whether that direction tracks the concept itself or only the words and tone that usually travel with it. The stress lists exist for the second job, so they must separate the concept from its surface cues.

Rules for every example:
- One or two sentences, 8 to 30 words, plain prose. No numbering, no labels, no quotation marks around the whole example, straight quotes and apostrophes only.
- Vary the domain (work, school, cooking, travel, health, hobbies, money, weather, family, technology, sports, nature, and so on) and vary the sentence openings. Never reuse an example, or a near paraphrase of one, in another list.
- Invent every name and organization. Never name a real person, company, product, institution, or event.

Tone-decoupling rule, which applies to every concept: wherever affect (warmth, enthusiasm, negativity, politeness, excitement) could stand in for the concept, the stress lists pull them apart. Implicit positives carry the concept in a cold or flat tone. Decoys carry the warm or emotional tone and the keywords, but not the concept. The training and held-out lists do not need tone matching: they represent the naive dataset a practitioner would build, and the stress lists show what that naivety costs. As an illustration, if the concept were sycophancy: an implicit positive is a flat, unadorned agreement with a false claim; a decoy is a warm, enthusiastic correction, or warm agreement with a claim that is actually true.

Response-property concepts: a concept is a response property when it describes how an assistant's reply relates to the user's message, so that judging it requires seeing that message (sycophancy, refusal, hedging, deception, verbosity, evasiveness); an emotion, tone, register, language, or topic is judged from the text alone and is not one. For a response property, every example in every list, including neutral and background, has a context (the user message, 8 to 30 words) and a text (the assistant reply). Positives are replies that show the property toward their context; negatives are replies to the same kind of context that do not; neutral and background examples are ordinary replies to mundane requests. A positive and its matched negative may share a context, but no text is ever repeated. For any other concept (a topic, style, register, emotion, language, or content property of a text on its own), every context is null."""

PLAN_PROMPT = """Concept: "{concept}"

Plan the contrast set for this concept. Do not write examples yet; four writers will each produce part of the set from this plan in parallel.

is_response_property: true only if the concept describes how an assistant's reply relates to the user's message, so that judging it requires seeing that message (sycophancy, refusal, hedging, deception, verbosity, evasiveness). A property that can be judged from a text on its own (an emotion, a tone such as sarcasm, a register, a language, a topic) is false, even though an assistant's replies can carry it.

keywords: 5 to 12 words, stems, or short phrases that a naive keyword detector would key on to spot the concept. Use content words and distinctive phrases, never articles, function words, or abbreviations: a keyword must not be the start of a common word in another sense or another language ("les" and "est" are wrong for French; "bonjour" and "merci" are right). Prefer stems where they are unambiguous ("apolog" covers apologize and apology). Every keyword check is a case-insensitive match at the start of a word.

tone_axis: one or two sentences naming the affect (warmth, enthusiasm, negativity, politeness, excitement, or none) that could stand in for this concept, and how the stress lists decouple it: implicit positives cold or flat, decoys warm or emotional. If no affect is entangled with the concept, say so.

confound_axis: a one- or two-word name for that tone or style axis ("warmth" for sycophancy, "negativity" for sarcasm, "formality" for legal language), or null if nothing is entangled with the concept.

train_topics: 32 distinct everyday topics, one per matched positive and negative pair in the training lists; the first 16 go to the explicit positives and the last 16 to the implicit ones. heldout_topics: 6 distinct topics not in train_topics. stress_topics: 12 distinct topics in neither list, for the implicit positives, decoys, and neutral examples. Spread all three lists across domains (work, school, cooking, travel, health, hobbies, money, weather, family, technology, sports, nature, and so on).

system_prompts: exactly 2. The first is a system prompt that makes an assistant's replies show the concept strongly; the second is one that suppresses it. Each is 1 to 3 sentences of plain instructions to an assistant. Describe the behavior you want rather than just naming the concept.

demo_user_message: one user message for a canned conversation with that assistant, 8 to 40 words. For a response-property concept, a confident false or dubious claim that invites the property. Otherwise, a request whose natural answer would invite the concept."""

PLAN_RETRY_PROMPT = """The plan failed these checks:
{violations}

Write the complete plan again, fixing these problems and keeping everything else."""

STAGE_HEADER = """Concept: "{concept}"

The plan for this contrast set is fixed. Follow it exactly: other writers are producing the remaining lists from the same plan in parallel, so use only your assigned topics and only the plan's keywords.
{plan_json}

{property_clause}

Write only the lists below.

"""

RESPONSE_PROPERTY_CLAUSE = "The plan marks this concept as a response property: every example has a context (the user message, 8 to 30 words) and a text (the assistant reply). A matched negative replies to the same context as its positive. Neutral and background examples are ordinary replies to mundane user messages."
TEXT_PROPERTY_CLAUSE = "The plan marks this concept as a property of the text itself: set every context to null."

TRAIN_NEG_RULE = "train_neg: 16 examples where the concept is clearly absent, matched one for one to train_pos in topic and length (item 3 of train_neg mirrors item 3 of train_pos), written in the natural, plain way. Do not match tone."

TRAIN_EXPLICIT_PROMPT = STAGE_HEADER + """Your topics, one per matched pair, in order: {topics}

train_pos: 16 explicit examples where the concept is clearly present and each uses at least one keyword from the plan, one per topic in order.

""" + TRAIN_NEG_RULE

TRAIN_IMPLICIT_PROMPT = STAGE_HEADER + """Your topics, one per matched pair, in order: {topics}

train_pos: 16 implicit examples where the concept is unmistakable but none of the plan's keywords appear in any form, one per topic in order. Another writer covers the keyword-bearing positives, so keep the keywords out entirely here.

""" + TRAIN_NEG_RULE

STRESS_PROMPT = STAGE_HEADER + """heldout_pos: 6 fresh positives, mixing explicit and implicit, one per heldout_topics entry. heldout_neg: 6 fresh negatives matched to them one for one in topic and length, tone not matched.

implicit_pos: 8 examples where the concept is clearly present, no keyword appears in any form, and the tone is cold or flat wherever the plan's tone axis could stand in for the concept. Use the stress_topics.

decoys: 16 examples that each contain at least one keyword verbatim while the concept is clearly absent. Spread these mechanisms across the 16: negation, quotation or reported speech, another sense of the word, meta-mention (talking about the word or the concept itself), and warm or emotional tone wherever the plan's tone axis could stand in for the concept. Use the stress_topics.

neutral: 6 examples of unrelated everyday text with no keywords and nothing to do with the concept: instructions, schedules, descriptions, small talk."""

BACKGROUND_PROMPT = STAGE_HEADER + """background: 24 examples in the same style and length as positives for this concept would have, on unrelated everyday topics that appear in none of the plan's topic lists, with the concept absent and no keywords. This is a broad pool, not the opposite of the concept: spread the topics and registers widely so that no single alternative concept dominates it (not all cheerful, not all instructions, not all one subject). For a response-property concept these are 24 ordinary helpful replies to varied user messages, each with its context, neither showing the property nor pushing against it."""

AXIS_PROMPT = """Concept: "{concept}"
{axis_line}
Keywords a naive detector uses for this concept: {keywords}.
{property_line}

Write pairs for fitting a direction that tracks the confound axis rather than the concept. Each pair is two versions of the same content with only the axis flipped: high carries the axis at full strength (for warmth: warm and effusive; for formality: highly formal), low carries its opposite (cold and flat; casual). The two texts state identical facts, take the same stance, cover the same content in the same order, and have the same word count within three words: the low version keeps the length by saying the same things in neutral, matter-of-fact wording, never by dropping content. Express the axis through word choice, intensifiers, and how the reader is addressed, spread across the whole text, not only through an added opening exclamation.

absent_pairs: 12 pairs where the concept is clearly absent in both texts.
present_pairs: 12 pairs where the concept is clearly present in both texts, equally strongly in high and in low.

Keywords may appear in either text of any pair; these pairs fit the axis, not the concept. Vary the domain from pair to pair. Every text is one or two sentences, 8 to 30 words. Set axis to the axis name."""

AXIS_GIVEN_LINE = 'Confound axis: "{axis}", the tone or style that the stress lists of this concept decouple.'
AXIS_OPEN_LINE = "First name the dominant tone or style axis that could stand in for this concept (warmth for sycophancy, formality for legal language), as one or two words in axis, then write the pairs along it."
AXIS_RESPONSE_PROPERTY_LINE = "This concept is a response property: each pair has a context (the user message, 8 to 30 words) and both texts reply to it. In present pairs both replies show the property toward that context (for sycophancy: both capitulate to the same false claim, one warmly and one flatly)."
AXIS_TEXT_PROPERTY_LINE = "This concept is a property of the text itself: set every context to null."

MINE_PROMPT = """Concept: "{concept}"
Keywords a naive detector uses for this concept: {keywords}.
Seed tokens: {seeds}. A probe trained for this concept fires on these tokens even when the concept is absent; they are its vehicle, and these examples exist to teach it otherwise.
{property_line}

Write 16 examples where the concept is clearly absent and each uses at least one seed token verbatim (the token may start a longer word). Spread these mechanisms across the 16: negation, quotation or reported speech, another sense of the word, meta-mention (talking about the word or the concept itself), and warm or emotional tone wherever affect could stand in for the concept. Vary the domain from example to example and rotate through the seeds rather than leaning on one. Every text is one or two sentences, 8 to 30 words."""

MINE_RESPONSE_PROPERTY_LINE = "This concept is a response property: each example has a context (the user message, 8 to 30 words) and the text is the assistant's reply, which must not show the property toward that message."
MINE_TEXT_PROPERTY_LINE = "This concept is a property of the text itself: set every context to null."

MINE_RETRY_PROMPT = """Automatic checks removed some of your examples, and the list is now short. Problems:
{violations}

Write the list again in full. Keep every example that passed, replace the removed ones with examples that satisfy the rules, and keep the same quality everywhere else."""

AXIS_RETRY_PROMPT = """Automatic checks removed some of your pairs, and a list is now short. Problems:
{violations}

Write both lists again in full. Keep every pair that passed, replace the removed ones with pairs that satisfy the rules, and keep the same quality everywhere else."""

RETRY_PROMPT = """Automatic checks removed some of your examples, and these lists are now short. Problems:
{violations}

Write the same lists again in full. Keep every example that passed, replace the removed ones with examples that satisfy the rules and the plan, and keep the same quality everywhere else."""


@dataclass(frozen=True)
class StageCall:
    name: str
    schema: type[BaseModel]
    prompt: str
    lists: tuple[str, ...]
    topic_half: int | None = None  # train calls each take one half of the plan's train_topics


STAGE_CALLS = (
    StageCall("train_explicit", TrainLists, TRAIN_EXPLICIT_PROMPT, ("train_pos", "train_neg"), topic_half=0),
    StageCall("train_implicit", TrainLists, TRAIN_IMPLICIT_PROMPT, ("train_pos", "train_neg"), topic_half=1),
    StageCall("stress", StressLists, STRESS_PROMPT, ("heldout_pos", "heldout_neg", "implicit_pos", "decoys", "neutral")),
    StageCall("background", BackgroundList, BACKGROUND_PROMPT, ("background",)),
)


class GenerationError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class Attempt:
    parsed: BaseModel
    raw_text: str
    wall_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class Validated:
    contrast_set: ContrastSet
    dropped: dict[str, int]
    violations: dict[str, str]  # list or field name -> description
    shortfall: int
    rejected: dict[str, list[tuple[str, str]]]  # list name -> (text, reason)


@dataclass
class Timings:
    plan_s: float = 0.0
    stage2_s: float = 0.0
    retry_s: float = 0.0
    total_s: float = 0.0
    calls: dict[str, float] = field(default_factory=dict)
    retried: list[str] = field(default_factory=list)


_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()
_counter_lock = threading.Lock()
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"'})


def normalize_concept(concept: str) -> str:
    """Strip, collapse whitespace, lowercase; reject empty, punctuation-only, or over-long input."""
    collapsed = re.sub(r"\s+", " ", concept.strip())
    if not collapsed or not any(ch.isalnum() for ch in collapsed):
        raise GenerationError(400, "Type a concept to probe for, such as sarcasm, legal language, or sadness.")
    if len(collapsed) > CONCEPT_MAX_CHARS:
        raise GenerationError(400, f"Keep the concept under {CONCEPT_MAX_CHARS} characters.")
    return collapsed.lower()


def generate_contrast_set(concept: str) -> ConceptResponse:
    normalized = normalize_concept(concept)
    cached = _read_cached(normalized)
    if cached is not None:
        return cached.model_copy(update={"cached": True})
    result, _ = _generate_uncached(normalized)
    return result


def _cache_dir() -> Path:
    return Path(DATA_DIR) / "cache"


def _set_path(normalized: str) -> Path:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return _cache_dir() / "sets" / f"{digest}.json"


def _counter_path() -> Path:
    return _cache_dir() / f"counter-{datetime.now(timezone.utc):%Y-%m-%d}.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_cached(normalized: str) -> ConceptResponse | None:
    path = _set_path(normalized)
    if not path.exists():
        return None
    try:
        return ConceptResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable cache entry %s: %s", path, exc)
        return None


def _reserve_generation() -> None:
    with _counter_lock:
        path = _counter_path()
        count = 0
        if path.exists():
            try:
                count = int(json.loads(path.read_text(encoding="utf-8")).get("count", 0))
            except (OSError, ValueError, AttributeError):
                count = 0
        if count >= DAILY_GENERATION_CAP:
            raise GenerationError(
                429,
                "Today's budget for new contrast sets is used up. The prebuilt examples still work, "
                "and new concepts open again tomorrow.",
            )
        _write_json(path, {"count": count + 1})


def _get_client() -> anthropic.Anthropic:
    global _client
    with _client_lock:
        if _client is None:
            load_dotenv(REPO_ROOT / ".env")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise GenerationError(500, "Generation is not configured on this server (no API key).")
            _client = anthropic.Anthropic(timeout=GEN_TIMEOUT_S, max_retries=0)  # a failed call surfaces at once; the validation retry is ours
        return _client


def _call_model(messages: list[dict[str, str]], *, model: str, effort: str, output_format: type[BaseModel]) -> Attempt:
    """One structured-output call; the parsed object plus its raw JSON text and timing.

    Streams so the SDK's read timeout measures gaps between chunks rather than the whole generation: a slow call
    finishes instead of timing out and being retried from scratch.
    """
    client = _get_client()
    started = time.monotonic()
    try:
        with client.messages.stream(
            model=model,
            max_tokens=GEN_MAX_TOKENS,
            output_config={"effort": effort},
            system=SYSTEM_PROMPT,
            messages=messages,
            output_format=output_format,
        ) as stream:
            response = stream.get_final_message()
    except pydantic.ValidationError as exc:
        log.warning("%s did not parse: %s", output_format.__name__, str(exc)[:500])
        raise GenerationError(502, "The model's output was cut off or malformed before it finished. Try again.") from exc
    except anthropic.APIConnectionError as exc:
        log.warning("could not reach the generation service: %s", exc)
        raise GenerationError(502, "Could not reach the generation service. Try again in a moment.") from exc
    except anthropic.APIStatusError as exc:
        log.warning("generation service error %s: %s", exc.status_code, exc.message)
        raise GenerationError(502, f"The generation service returned an error (HTTP {exc.status_code}). Try again in a moment.") from exc
    except anthropic.APIError as exc:
        log.warning("generation service error: %s", exc.message)
        raise GenerationError(502, "The generation service returned an error. Try again in a moment.") from exc
    if response.stop_reason == "refusal":
        raise GenerationError(
            400,
            "The model declined to write examples for this concept. Try a different concept, or explore the prebuilt examples.",
        )
    parsed = response.parsed_output
    if parsed is None:
        if response.stop_reason == "max_tokens":
            raise GenerationError(502, "The set ran past the output limit before it finished. Try again.")
        raise GenerationError(502, "The model returned no contrast set. Try again.")
    raw_text = next((block.text for block in response.content if block.type == "text"), parsed.model_dump_json())
    attempt = Attempt(parsed, raw_text, time.monotonic() - started, response.usage.input_tokens, response.usage.output_tokens)
    log.info(
        "%s via %s/%s in %.1fs: %d input tokens, %d output tokens (cap %d), stop_reason=%s",
        output_format.__name__, model, effort, attempt.wall_s, attempt.input_tokens, attempt.output_tokens,
        GEN_MAX_TOKENS, response.stop_reason,
    )
    return attempt


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value.translate(_QUOTE_MAP)).strip()


def _clean_list(values: list[str]) -> list[str]:
    return [v for v in (_clean(v) or "" for v in values) if v]


def _clean_keywords(keywords: list[str]) -> list[str]:
    return [k for k in dict.fromkeys(_clean_list(keywords)) if len(k) >= MIN_KEYWORD_CHARS]


def _matching_keywords(text: str, keywords_lower: list[str]) -> list[str]:
    """Case-insensitive prefix-stem match: the keyword must start at a word boundary ("sad" hits "sadness", not "otherwise" for "wise")."""
    lowered = text.lower()
    return [k for k in keywords_lower if re.search(r"\b" + re.escape(k), lowered)]


def _fallback_system_prompts(concept: str) -> list[str]:
    return [
        f"You are an assistant whose replies strongly show {concept}. Let it shape every answer you give.",
        f"You are an assistant whose replies never show {concept}. Keep every answer free of it.",
    ]


def _fallback_demo_message(concept: str) -> str:
    return f"Can you talk me through something in a way that shows {concept}?"


def _clean_axis_name(value: str | None) -> str | None:
    cleaned = (_clean(value) or "").strip(" .").lower()
    return cleaned or None


def _clean_plan(plan: GenerationPlan) -> GenerationPlan:
    return plan.model_copy(
        update={
            "keywords": _clean_keywords(plan.keywords),
            "tone_axis": _clean(plan.tone_axis) or "",
            "confound_axis": _clean_axis_name(plan.confound_axis),
            "train_topics": _clean_list(plan.train_topics),
            "heldout_topics": _clean_list(plan.heldout_topics),
            "stress_topics": _clean_list(plan.stress_topics),
            "system_prompts": _clean_list(plan.system_prompts),
            "demo_user_message": _clean(plan.demo_user_message) or "",
        }
    )


def _plan_violations(plan: GenerationPlan) -> dict[str, str]:
    violations: dict[str, str] = {}
    if not plan.keywords:
        violations["keywords"] = f"keywords: none survived (each must be at least {MIN_KEYWORD_CHARS} characters)"
    if len(plan.system_prompts) != 2:
        violations["system_prompts"] = f"system_prompts: got {len(plan.system_prompts)}, need exactly 2 (raise first, then lower)"
    if not plan.demo_user_message:
        violations["demo_user_message"] = "demo_user_message: empty"
    return violations


def _finalize_plan(plan: GenerationPlan, concept: str) -> GenerationPlan:
    """Fill the fields stage 2 depends on so every writer sees a complete plan."""
    prompts = plan.system_prompts
    return plan.model_copy(
        update={
            "system_prompts": (prompts + _fallback_system_prompts(concept)[len(prompts):])[:2],
            "demo_user_message": plan.demo_user_message or _fallback_demo_message(concept),
        }
    )


def _reject_reason(
    list_name: str,
    text: str,
    context: str | None,
    keywords_lower: list[str],
    is_response_property: bool,
    seen: set[str],
) -> str | None:
    if not text:
        return "empty text"
    words = len(text.split())
    if words < MIN_WORDS:
        return f"under {MIN_WORDS} words"
    if words > MAX_WORDS:
        return f"over {MAX_WORDS} words"
    if is_response_property and not context:
        return "missing context"
    if text.lower() in seen:
        return "duplicate of an earlier example"
    if list_name in ("implicit_pos", "background"):
        hits = _matching_keywords(text, keywords_lower)
        if hits:
            return "contains keyword " + ", ".join(hits)
    if list_name == "decoys" and not _matching_keywords(text, keywords_lower):
        return "contains no keyword"
    return None


def _validate(raw: ContrastSet, concept: str) -> Validated:
    keywords = _clean_keywords(raw.keywords)
    keywords_lower = [k.lower() for k in keywords]
    is_rp = raw.is_response_property
    seen: set[str] = set()
    kept: dict[str, list[Example]] = {}
    dropped: dict[str, int] = {}
    rejected: dict[str, list[tuple[str, str]]] = {}
    for name in LIST_TARGETS:
        survivors: list[Example] = []
        for example in getattr(raw, name):
            text = _clean(example.text) or ""
            context = _clean(example.context) if is_rp else None
            reason = _reject_reason(name, text, context, keywords_lower, is_rp, seen)
            if reason is not None:
                rejected.setdefault(name, []).append((text, reason))
                continue
            seen.add(text.lower())
            survivors.append(Example(text=text, context=context or None))
        if len(survivors) > LIST_SIZES[name]:
            log.info("%s: truncating %d survivors to %d", name, len(survivors), LIST_SIZES[name])
        kept[name] = survivors[: LIST_SIZES[name]]
        dropped[name] = max(0, LIST_SIZES[name] - len(kept[name]))

    violations: dict[str, str] = {}
    shortfall = 0
    for name, minimum in LIST_MINIMUMS.items():
        if len(kept[name]) < minimum:
            shortfall += minimum - len(kept[name])
            violations[name] = (
                f"{name}: only {len(kept[name])} of {LIST_TARGETS[name]} survived, need at least {minimum}. "
                f"Removed: {_describe_rejects(rejected.get(name, []))}"
            )
    system_prompts = _clean_list(raw.system_prompts)
    if len(system_prompts) != 2:
        shortfall += 1
        violations["system_prompts"] = f"system_prompts: got {len(system_prompts)}, need exactly 2 (raise first, then lower)"
    demo = _clean(raw.demo_user_message) or ""
    if not demo:
        shortfall += 1
        violations["demo_user_message"] = "demo_user_message: empty"
    if not keywords:
        shortfall += 1
        violations["keywords"] = f"keywords: none survived (each must be at least {MIN_KEYWORD_CHARS} characters)"

    contrast_set = ContrastSet(
        concept=concept,
        is_response_property=is_rp,
        keywords=keywords,
        confound_axis=_clean_axis_name(raw.confound_axis),
        system_prompts=(system_prompts + _fallback_system_prompts(concept)[len(system_prompts):])[:2],
        demo_user_message=demo or _fallback_demo_message(concept),
        **kept,
    )
    return Validated(contrast_set, dropped, violations, shortfall, rejected)


def _describe_rejects(rejects: list[tuple[str, str]]) -> str:
    return "; ".join(f'"{text[:80]}" ({reason})' for text, reason in rejects[:6]) or "fewer than requested were generated"


def _merge(concept: str, plan: GenerationPlan, parts: dict[str, BaseModel]) -> ContrastSet:
    lists: dict[str, list[Example]] = {}
    for call in STAGE_CALLS:
        for name in call.lists:
            lists.setdefault(name, []).extend(getattr(parts[call.name], name))
    return ContrastSet(
        concept=concept,
        is_response_property=plan.is_response_property,
        keywords=plan.keywords,
        confound_axis=plan.confound_axis,
        system_prompts=plan.system_prompts,
        demo_user_message=plan.demo_user_message,
        **lists,
    )


def _format_violations(violations: dict[str, str]) -> str:
    return "\n".join(f"- {v}" for v in violations.values())


def _plan(normalized: str, timings: Timings, *, model: str, effort: str) -> GenerationPlan:
    started = time.monotonic()
    messages: list[dict[str, str]] = [{"role": "user", "content": PLAN_PROMPT.format(concept=normalized)}]
    attempt = _call_model(messages, model=model, effort=effort, output_format=GenerationPlan)
    plan = _clean_plan(attempt.parsed)  # type: ignore[arg-type]
    violations = _plan_violations(plan)
    if violations:
        log.warning("retrying plan for %r once; violations: %s", normalized, list(violations.values()))
        retry = _call_model(
            messages
            + [
                {"role": "assistant", "content": attempt.raw_text},
                {"role": "user", "content": PLAN_RETRY_PROMPT.format(violations=_format_violations(violations))},
            ],
            model=model,
            effort=effort,
            output_format=GenerationPlan,
        )
        second = _clean_plan(retry.parsed)  # type: ignore[arg-type]
        if len(_plan_violations(second)) <= len(violations):
            plan = second
    timings.plan_s = time.monotonic() - started
    plan = _finalize_plan(plan, normalized)
    log.info(
        "plan for %r in %.1fs: response_property=%s, keywords=%s, topics=%d/%d/%d",
        normalized, timings.plan_s, plan.is_response_property, plan.keywords,
        len(plan.train_topics), len(plan.heldout_topics), len(plan.stress_topics),
    )
    return plan


def _stage_messages(normalized: str, plan: GenerationPlan, call: StageCall) -> list[dict[str, str]]:
    clause = RESPONSE_PROPERTY_CLAUSE if plan.is_response_property else TEXT_PROPERTY_CLAUSE
    topics = ""
    if call.topic_half is not None:
        half = len(plan.train_topics) // 2
        chosen = plan.train_topics[:half] if call.topic_half == 0 else plan.train_topics[half:]
        topics = "; ".join(chosen) or f"choose {TRAIN_HALF_SIZE} varied everyday topics"
    prompt = call.prompt.format(concept=normalized, plan_json=plan.model_dump_json(indent=1), property_clause=clause, topics=topics)
    return [{"role": "user", "content": prompt}]


def _call_violations(best: Validated, parts: dict[str, BaseModel]) -> dict[str, dict[str, str]]:
    """Violations attributed to the stage-2 call that must fix them. Train halves are judged on their own survivors."""
    failing: dict[str, dict[str, str]] = {}
    for call in STAGE_CALLS:
        if call.topic_half is None:
            found = {name: best.violations[name] for name in call.lists if name in best.violations}
            if found:
                failing[call.name] = found
            continue
        for name in call.lists:
            own = {(_clean(e.text) or "").lower() for e in getattr(parts[call.name], name)}
            kept = sum(1 for e in getattr(best.contrast_set, name) if e.text.lower() in own)
            if kept < TRAIN_HALF_MINIMUM:
                rejects = [(text, reason) for text, reason in best.rejected.get(name, []) if text.lower() in own]
                failing.setdefault(call.name, {})[name] = (
                    f"{name}: only {kept} of your {TRAIN_HALF_SIZE} survived, need at least {TRAIN_HALF_MINIMUM}. "
                    f"Removed: {_describe_rejects(rejects)}"
                )
    return failing


def _total_shortfall(best: Validated, parts: dict[str, BaseModel]) -> int:
    per_half = 0
    for call in STAGE_CALLS:
        if call.topic_half is None:
            continue
        for name in call.lists:
            own = {(_clean(e.text) or "").lower() for e in getattr(parts[call.name], name)}
            kept = sum(1 for e in getattr(best.contrast_set, name) if e.text.lower() in own)
            per_half += max(0, TRAIN_HALF_MINIMUM - kept)
    return best.shortfall + per_half


def _generate_uncached(
    normalized: str,
    *,
    model: str = GEN_MODEL,
    effort: str = GEN_EFFORT,
    plan_model: str = GEN_PLAN_MODEL,
    plan_effort: str = GEN_PLAN_EFFORT,
) -> tuple[ConceptResponse, Timings]:
    _reserve_generation()
    timings = Timings()
    started = time.monotonic()
    plan = _plan(normalized, timings, model=plan_model, effort=plan_effort)

    stage_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(STAGE_CALLS)) as pool:
        futures = {
            call.name: pool.submit(
                _call_model, _stage_messages(normalized, plan, call), model=model, effort=effort, output_format=call.schema
            )
            for call in STAGE_CALLS
        }
        attempts = {name: future.result() for name, future in futures.items()}
    timings.stage2_s = time.monotonic() - stage_started
    timings.calls = {name: round(attempt.wall_s, 1) for name, attempt in attempts.items()}
    parts = {name: attempt.parsed for name, attempt in attempts.items()}
    best = _validate(_merge(normalized, plan, parts), normalized)
    log.info("stage 2 for %r in %.1fs (calls %s): dropped=%s", normalized, timings.stage2_s, timings.calls, best.dropped)

    failing = _call_violations(best, parts)
    if failing:
        retry_started = time.monotonic()
        timings.retried = list(failing)
        log.warning("retrying %s for %r; violations: %s", timings.retried, normalized, failing)
        calls_by_name = {call.name: call for call in STAGE_CALLS}
        with ThreadPoolExecutor(max_workers=len(failing)) as pool:
            futures = {
                name: pool.submit(
                    _call_model,
                    _stage_messages(normalized, plan, calls_by_name[name])
                    + [
                        {"role": "assistant", "content": attempts[name].raw_text},
                        {"role": "user", "content": RETRY_PROMPT.format(violations=_format_violations(violations))},
                    ],
                    model=model,
                    effort=effort,
                    output_format=calls_by_name[name].schema,
                )
                for name, violations in failing.items()
            }
            retries = {name: future.result() for name, future in futures.items()}
        for name, attempt in retries.items():
            candidate_parts = {**parts, name: attempt.parsed}
            candidate = _validate(_merge(normalized, plan, candidate_parts), normalized)
            if _total_shortfall(candidate, candidate_parts) <= _total_shortfall(best, parts):
                parts, best = candidate_parts, candidate
        timings.retry_s = time.monotonic() - retry_started
        log.info("after retry for %r: dropped=%s, violations=%s", normalized, best.dropped, list(best.violations.values()))

    timings.total_s = time.monotonic() - started
    if timings.total_s > SLOW_GENERATION_S:
        log.warning(
            "generation for %r took %.0fs, over the %ds budget (plan %.0fs, stage 2 %.0fs, retry %.0fs)",
            normalized, timings.total_s, SLOW_GENERATION_S, timings.plan_s, timings.stage2_s, timings.retry_s,
        )
    if best.violations:
        log.warning("returning %r with unresolved violations: %s", normalized, list(best.violations.values()))
    result = ConceptResponse(contrast_set=best.contrast_set, dropped=best.dropped, cached=False)
    _write_json(_set_path(normalized), result.model_dump(mode="json"))
    return result, timings


def generate_axis_set(concept: str, contrast_set: ContrastSet) -> AxisResponse:
    normalized = normalize_concept(concept)
    cached = _read_cached_axis(_axis_path(normalized, contrast_set))
    if cached is not None:
        return cached.model_copy(update={"cached": True})
    result, _ = _generate_axis_uncached(normalized, contrast_set)
    return result


def _axis_path(normalized: str, contrast_set: ContrastSet) -> Path:
    axis = _clean_axis_name(contrast_set.confound_axis) or ""
    key = "\n".join([normalized, axis, *_clean_keywords(contrast_set.keywords)])
    return _cache_dir() / "axis" / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _read_cached_axis(path: Path) -> AxisResponse | None:
    if not path.exists():
        return None
    try:
        return AxisResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable cache entry %s: %s", path, exc)
        return None


@dataclass
class ValidatedAxis:
    axis_set: AxisSet
    dropped: dict[str, int]
    violations: dict[str, str]
    shortfall: int


def _validate_axis(raw: AxisSet, axis_name: str | None, is_response_property: bool) -> ValidatedAxis:
    kept: dict[str, list[AxisPair]] = {}
    dropped: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for name, size in AXIS_LIST_TARGETS.items():
        survivors: list[AxisPair] = []
        for pair in getattr(raw, name):
            high = _clean(pair.high) or ""
            low = _clean(pair.low) or ""
            context = _clean(pair.context) if is_response_property else None
            reason = None
            if not high or not low:
                reason = "empty text"
            elif min(len(high.split()), len(low.split())) < MIN_WORDS:
                reason = f"under {MIN_WORDS} words"
            elif max(len(high.split()), len(low.split())) > MAX_WORDS:
                reason = f"over {MAX_WORDS} words"
            elif high.lower() == low.lower():
                reason = "high and low are identical"
            elif abs(len(high.split()) - len(low.split())) > AXIS_MAX_LENGTH_GAP:
                reason = f"length gap of {abs(len(high.split()) - len(low.split()))} words (limit {AXIS_MAX_LENGTH_GAP})"
            elif is_response_property and not context:
                reason = "missing context"
            if reason is not None:
                reasons.setdefault(name, []).append(f'"{high[:60]}" / "{low[:60]}" ({reason})')
                continue
            survivors.append(AxisPair(high=high, low=low, context=context or None))
        if len(survivors) > size:
            log.info("%s: truncating %d survivors to %d", name, len(survivors), size)
        kept[name] = survivors[:size]
        dropped[name] = max(0, size - len(kept[name]))

    violations: dict[str, str] = {}
    shortfall = 0
    for name, size in AXIS_LIST_TARGETS.items():
        if len(kept[name]) < AXIS_LIST_MINIMUM:
            shortfall += AXIS_LIST_MINIMUM - len(kept[name])
            removed = "; ".join(reasons.get(name, [])[:6]) or "fewer than requested were generated"
            violations[name] = f"{name}: only {len(kept[name])} of {size} survived, need at least {AXIS_LIST_MINIMUM}. Removed: {removed}"
    axis = axis_name or _clean_axis_name(raw.axis) or "tone"
    return ValidatedAxis(AxisSet(axis=axis, **kept), dropped, violations, shortfall)


def _generate_axis_uncached(
    normalized: str, contrast_set: ContrastSet, *, model: str = GEN_MODEL, effort: str = GEN_EFFORT
) -> tuple[AxisResponse, float]:
    _reserve_generation()
    started = time.monotonic()
    axis_name = _clean_axis_name(contrast_set.confound_axis)
    keywords = _clean_keywords(contrast_set.keywords)
    prompt = AXIS_PROMPT.format(
        concept=normalized,
        axis_line=AXIS_GIVEN_LINE.format(axis=axis_name) if axis_name else AXIS_OPEN_LINE,
        keywords=", ".join(keywords) or "none",
        property_line=AXIS_RESPONSE_PROPERTY_LINE if contrast_set.is_response_property else AXIS_TEXT_PROPERTY_LINE,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    attempt = _call_model(messages, model=model, effort=effort, output_format=AxisSet)
    best = _validate_axis(attempt.parsed, axis_name, contrast_set.is_response_property)  # type: ignore[arg-type]
    log.info("axis pairs for %r (%s): dropped=%s", normalized, best.axis_set.axis, best.dropped)
    if best.violations:
        log.warning("retrying axis pairs for %r once; violations: %s", normalized, list(best.violations.values()))
        retry = _call_model(
            messages
            + [
                {"role": "assistant", "content": attempt.raw_text},
                {"role": "user", "content": AXIS_RETRY_PROMPT.format(violations=_format_violations(best.violations))},
            ],
            model=model,
            effort=effort,
            output_format=AxisSet,
        )
        second = _validate_axis(retry.parsed, axis_name, contrast_set.is_response_property)  # type: ignore[arg-type]
        if second.shortfall <= best.shortfall:
            best = second
        log.info("after axis retry for %r: dropped=%s, violations=%s", normalized, best.dropped, list(best.violations.values()))
    wall = time.monotonic() - started
    if best.violations:
        log.warning("returning axis pairs for %r with unresolved violations: %s", normalized, list(best.violations.values()))
    result = AxisResponse(axis_set=best.axis_set, dropped=best.dropped, cached=False)
    _write_json(_axis_path(normalized, contrast_set), result.model_dump(mode="json"))
    return result, wall


def _clean_seeds(seeds: list[str]) -> list[str]:
    """Strip whitespace and tokenizer word-start markers; dedupe case-insensitively; keep the caller's order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for seed in seeds:
        value = (_clean(seed) or "").lstrip("▁Ġ").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            cleaned.append(value)
    return cleaned


def generate_decoys_from_seeds(concept: str, contrast_set: ContrastSet, seeds: list[str]) -> MineResponse:
    normalized = normalize_concept(concept)
    cleaned = _clean_seeds(seeds)
    if not cleaned:
        raise GenerationError(400, "Give at least one seed token to build decoys around.")
    cached = _read_cached_mine(_mine_path(normalized, cleaned))
    if cached is not None:
        return cached.model_copy(update={"cached": True})
    result, _ = _generate_mined_uncached(normalized, contrast_set, cleaned)
    return result


def _mine_path(normalized: str, seeds: list[str]) -> Path:
    key = "\n".join([normalized, *sorted(seed.lower() for seed in seeds)])
    return _cache_dir() / "mined" / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _read_cached_mine(path: Path) -> MineResponse | None:
    if not path.exists():
        return None
    try:
        return MineResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable cache entry %s: %s", path, exc)
        return None


@dataclass
class ValidatedMine:
    decoys: list[Example]
    dropped: dict[str, int]
    violation: str | None
    shortfall: int


def _validate_mined(raw: MinedDecoys, seeds: list[str], is_response_property: bool) -> ValidatedMine:
    seeds_lower = [seed.lower() for seed in seeds]
    seen: set[str] = set()
    survivors: list[Example] = []
    rejects: list[tuple[str, str]] = []
    for example in raw.decoys:
        text = _clean(example.text) or ""
        context = _clean(example.context) if is_response_property else None
        reason = _reject_reason("decoys", text, context, seeds_lower, is_response_property, seen)
        if reason is not None:
            rejects.append((text, reason.replace("keyword", "seed")))
            continue
        seen.add(text.lower())
        survivors.append(Example(text=text, context=context or None))
    if len(survivors) > MINE_SIZE:
        log.info("mined decoys: truncating %d survivors to %d", len(survivors), MINE_SIZE)
    kept = survivors[:MINE_SIZE]
    violation = None
    shortfall = max(0, MINE_MINIMUM - len(kept))
    if shortfall:
        violation = f"decoys: only {len(kept)} of {MINE_TARGET} survived, need at least {MINE_MINIMUM}. Removed: {_describe_rejects(rejects)}"
    return ValidatedMine(kept, {"decoys": max(0, MINE_SIZE - len(kept))}, violation, shortfall)


def _generate_mined_uncached(
    normalized: str, contrast_set: ContrastSet, seeds: list[str], *, model: str = GEN_MODEL, effort: str = GEN_EFFORT
) -> tuple[MineResponse, float]:
    _reserve_generation()
    started = time.monotonic()
    prompt = MINE_PROMPT.format(
        concept=normalized,
        keywords=", ".join(_clean_keywords(contrast_set.keywords)) or "none",
        seeds=", ".join(f'"{seed}"' for seed in seeds),
        property_line=MINE_RESPONSE_PROPERTY_LINE if contrast_set.is_response_property else MINE_TEXT_PROPERTY_LINE,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    attempt = _call_model(messages, model=model, effort=effort, output_format=MinedDecoys)
    best = _validate_mined(attempt.parsed, seeds, contrast_set.is_response_property)  # type: ignore[arg-type]
    log.info("mined decoys for %r from %s: kept %d, dropped=%s", normalized, seeds, len(best.decoys), best.dropped)
    if best.violation:
        log.warning("retrying mined decoys for %r once; %s", normalized, best.violation)
        retry = _call_model(
            messages
            + [
                {"role": "assistant", "content": attempt.raw_text},
                {"role": "user", "content": MINE_RETRY_PROMPT.format(violations=f"- {best.violation}")},
            ],
            model=model,
            effort=effort,
            output_format=MinedDecoys,
        )
        second = _validate_mined(retry.parsed, seeds, contrast_set.is_response_property)  # type: ignore[arg-type]
        if second.shortfall <= best.shortfall:
            best = second
    wall = time.monotonic() - started
    if best.violation:
        log.warning("returning mined decoys for %r with unresolved violation: %s", normalized, best.violation)
    result = MineResponse(decoys=best.decoys, dropped=best.dropped, cached=False)
    _write_json(_mine_path(normalized, seeds), result.model_dump(mode="json"))
    return result, wall


def _load_contrast_set(path: Path) -> ContrastSet:
    text = path.read_text(encoding="utf-8")
    try:
        return ConceptResponse.model_validate_json(text).contrast_set
    except pydantic.ValidationError:
        return ContrastSet.model_validate_json(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a contrast set for a concept and write it as JSON.")
    parser.add_argument("concept")
    parser.add_argument("--out", type=Path, help="write the ConceptResponse JSON to this path")
    parser.add_argument("--fresh", action="store_true", help="skip the cache read and regenerate (the result still gets cached)")
    parser.add_argument("--model", default=GEN_MODEL, help="override GEN_MODEL for the example-writing calls (implies --fresh)")
    parser.add_argument("--effort", default=GEN_EFFORT, help="override GEN_EFFORT for the example-writing calls (implies --fresh)")
    parser.add_argument("--plan-model", default=GEN_PLAN_MODEL, help="override GEN_PLAN_MODEL for the plan call (implies --fresh)")
    parser.add_argument("--plan-effort", default=GEN_PLAN_EFFORT, help="override GEN_PLAN_EFFORT for the plan call (implies --fresh)")
    parser.add_argument("--axis", action="store_true", help="write confound-axis pairs for an existing set instead of a contrast set; needs --set")
    parser.add_argument("--mine", action="store_true", help="write concept-absent decoys around seed tokens instead of a contrast set; needs --set and --seeds")
    parser.add_argument("--seeds", default="", help="comma-separated seed tokens for --mine")
    parser.add_argument("--set", type=Path, help="ConceptResponse or ContrastSet JSON that --axis or --mine builds on")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    if args.axis or args.mine:
        if args.set is None:
            parser.error("--axis and --mine need --set <path to the concept's set JSON>")
        return _main_mine(args) if args.mine else _main_axis(args)

    started = time.monotonic()
    timings: Timings | None = None
    try:
        overridden = (args.model, args.effort, args.plan_model, args.plan_effort) != (GEN_MODEL, GEN_EFFORT, GEN_PLAN_MODEL, GEN_PLAN_EFFORT)
        if args.fresh or overridden:
            result, timings = _generate_uncached(
                normalize_concept(args.concept),
                model=args.model, effort=args.effort, plan_model=args.plan_model, plan_effort=args.plan_effort,
            )
        else:
            result = generate_contrast_set(args.concept)
    except GenerationError as exc:
        print(f"error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    wall = time.monotonic() - started
    if args.out:
        _write_json(args.out, result.model_dump(mode="json"))
    contrast_set = result.contrast_set
    summary = {
        "concept": contrast_set.concept,
        "model": args.model,
        "effort": args.effort,
        "plan_model": args.plan_model,
        "plan_effort": args.plan_effort,
        "wall_s": round(wall, 1),
        "timings": None if timings is None else {
            "plan_s": round(timings.plan_s, 1),
            "stage2_s": round(timings.stage2_s, 1),
            "calls": timings.calls,
            "retried": timings.retried,
            "retry_s": round(timings.retry_s, 1),
        },
        "cached": result.cached,
        "is_response_property": contrast_set.is_response_property,
        "keywords": contrast_set.keywords,
        "counts": {name: len(getattr(contrast_set, name)) for name in LIST_TARGETS},
        "dropped": result.dropped,
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _main_axis(args: argparse.Namespace) -> int:
    contrast_set = _load_contrast_set(args.set)
    started = time.monotonic()
    try:
        if args.fresh or (args.model, args.effort) != (GEN_MODEL, GEN_EFFORT):
            result, _ = _generate_axis_uncached(normalize_concept(args.concept), contrast_set, model=args.model, effort=args.effort)
        else:
            result = generate_axis_set(args.concept, contrast_set)
    except GenerationError as exc:
        print(f"error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    wall = time.monotonic() - started
    if args.out:
        _write_json(args.out, result.model_dump(mode="json"))
    summary = {
        "concept": normalize_concept(args.concept),
        "axis": result.axis_set.axis,
        "model": args.model,
        "effort": args.effort,
        "wall_s": round(wall, 1),
        "cached": result.cached,
        "is_response_property": contrast_set.is_response_property,
        "counts": {name: len(getattr(result.axis_set, name)) for name in AXIS_LIST_TARGETS},
        "dropped": result.dropped,
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _main_mine(args: argparse.Namespace) -> int:
    contrast_set = _load_contrast_set(args.set)
    seeds = [seed for seed in args.seeds.split(",")]
    started = time.monotonic()
    try:
        cleaned = _clean_seeds(seeds)
        if args.fresh or (args.model, args.effort) != (GEN_MODEL, GEN_EFFORT):
            if not cleaned:
                raise GenerationError(400, "Give at least one seed token to build decoys around.")
            result, _ = _generate_mined_uncached(normalize_concept(args.concept), contrast_set, cleaned, model=args.model, effort=args.effort)
        else:
            result = generate_decoys_from_seeds(args.concept, contrast_set, seeds)
    except GenerationError as exc:
        print(f"error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    wall = time.monotonic() - started
    if args.out:
        _write_json(args.out, result.model_dump(mode="json"))
    summary = {
        "concept": normalize_concept(args.concept),
        "seeds": cleaned,
        "model": args.model,
        "effort": args.effort,
        "wall_s": round(wall, 1),
        "cached": result.cached,
        "count": len(result.decoys),
        "dropped": result.dropped,
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
