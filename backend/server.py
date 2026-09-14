"""Probe Bench API. Run from the repo root: ``uvicorn backend.server:app --host 0.0.0.0 --port 8000``."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers

from backend import config
from backend.contract import (
    AxisRequest,
    AxisResponse,
    ChatRequest,
    ChatScored,
    ConceptRequest,
    ConceptResponse,
    ContrastSet,
    DeflateRequest,
    DeflateResponse,
    FitRequest,
    FitResponse,
    Health,
    MineRequest,
    MineResponse,
    MisfiresRequest,
    MisfiresResponse,
    ProbeInfo,
    ScoredTurn,
    ScoreRequest,
    ScoreResponse,
)
from backend.model import REPLACEMENT_CHAR, Runner, Span, flags_for, token_strings
from backend.probe import Probe, ProbeStore, deflate_probe, fit_probe, misfires

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe-bench")

try:
    from backend.generate import GenerationError, generate_contrast_set
except Exception:  # the generator module is optional at boot; /concept answers 503 without it
    log.warning("backend.generate unavailable; POST /concept will return 503", exc_info=True)
    GenerationError = None
    generate_contrast_set = None
try:
    from backend.generate import generate_axis_set
except Exception:  # same for the axis-pair generator; /axis answers 503 without it
    log.warning("backend.generate.generate_axis_set unavailable; POST /axis will return 503", exc_info=True)
    generate_axis_set = None
try:
    from backend.generate import generate_decoys_from_seeds
except Exception:  # same for the decoy miner; /mine answers 503 without it
    log.warning("backend.generate.generate_decoys_from_seeds unavailable; POST /mine will return 503", exc_info=True)
    generate_decoys_from_seeds = None

RUNNER = Runner()
STORE = ProbeStore()
ALPHA_LIMIT = 3.0
KEEPALIVE_SECONDS = 15
CONCEPT_HEARTBEAT_SECONDS = 10
MIN_REPLY_TOKENS = 32  # below this much headroom under the conversation cap, ask for a reset instead of a stub reply
MAX_MESSAGES = 80
MAX_DEFLATE_ROUNDS = 4
MAX_CONVERSATION_CHARS = 2 * config.MAX_CONVERSATION_TOKENS * 4  # rejected before rendering; a token is rarely under 2 chars
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_line(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def example_count(cs: ContrastSet) -> int:
    return sum(len(lst) for lst in (cs.train_pos, cs.train_neg, cs.heldout_pos, cs.heldout_neg, cs.implicit_pos, cs.decoys, cs.neutral, cs.background))


def require_within_fit_cap(n: int, what: str) -> None:
    if n > config.MAX_EXAMPLES_PER_FIT:
        raise HTTPException(400, f"at most {config.MAX_EXAMPLES_PER_FIT} {what} per fit ({n} given)")


async def run_fit(fn, *args):
    """Run a fit in the threadpool. Bad input is a 400; anything else is a JSON 500, never Starlette's plain text."""
    try:
        return await run_in_threadpool(fn, *args)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("%s failed", getattr(fn, "__name__", "fit"))
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=RUNNER.load, name="model-load", daemon=True).start()
    yield


class PasswordMiddleware:
    """Refuses any request whose X-Probe-Key header is not the demo password, before a route (or a stream)
    starts. GET /health stays open; OPTIONS passes so CORS preflights never need the key."""

    OPEN_PATHS = frozenset({"/health"})

    def __init__(self, app, password: str) -> None:
        self.app = app
        self._password = password.encode("utf-8")

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["method"] != "OPTIONS" and scope["path"] not in self.OPEN_PATHS:
            supplied = Headers(scope=scope).get("x-probe-key", "").encode("utf-8")
            if not hmac.compare_digest(supplied, self._password):
                await JSONResponse({"error": "password required"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = FastAPI(title="Probe Bench API", lifespan=lifespan)
if config.DEMO_PASSWORD:
    app.add_middleware(PasswordMiddleware, password=config.DEMO_PASSWORD)
# Added last, so outermost: preflights are answered here and 401s carry CORS headers the browser can read.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log.info("access password %s", "required on every route but /health" if config.DEMO_PASSWORD else "not set; the server is open")


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return JSONResponse({"error": f"invalid request: {where}: {first.get('msg', 'validation failed')}"}, status_code=400)


def require_model() -> None:
    if not RUNNER.loaded:
        raise HTTPException(503, "the model is still loading, try again in a minute" if RUNNER.loading else "the model is not loaded")


def require_probe(probe_id: str) -> Probe:
    probe = STORE.get(probe_id)
    if probe is None:
        raise HTTPException(404, f"probe {probe_id!r} not found; refit from the contrast set")
    if probe.n_layers != RUNNER.n_layers:
        raise HTTPException(409, "this probe was fit on a different model; refit from the contrast set")
    return probe


# ------------------------------------------------------------------ routes


@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(model_loaded=RUNNER.loaded, loading=RUNNER.loading, model_id=config.MODEL_ID, n_layers=RUNNER.n_layers)


@app.get("/auth")
async def auth() -> dict:
    """Only reachable with a valid X-Probe-Key when a password is set, so a client can check one before storing it."""
    return {"ok": True}


def generation_stream(fn, args: tuple, event_type: str, stage: str) -> StreamingResponse:
    """Generation can outlast the proxy's 100 s first-byte limit, so the result arrives as server-sent
    events: a status heartbeat immediately and every 10 s, then one ``event_type`` event (or one error)."""
    task = asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def sse():
        started = time.monotonic()
        yield sse_line({"type": "status", "text": "checking the cache"})
        while True:
            done, _ = await asyncio.wait({task}, timeout=CONCEPT_HEARTBEAT_SECONDS)
            if done:
                break
            elapsed = int(time.monotonic() - started)
            note = ", this can take a few minutes" if elapsed >= 60 else ""
            yield sse_line({"type": "status", "text": f"generating {stage} ({elapsed} s{note})"})
        try:
            result = task.result()
        except GenerationError as e:
            yield sse_line({"type": "error", "error": e.message, "status": e.status_code})
            return
        except Exception as e:
            log.exception("%s generation failed", event_type)
            yield sse_line({"type": "error", "error": f"{type(e).__name__}: {e}", "status": 500})
            return
        yield sse_line({"type": event_type, **result.model_dump()})

    return StreamingResponse(sse(), media_type="text/event-stream", headers=SSE_HEADERS)


async def generation_json(fn, args: tuple):
    """Plain JSON form of a generation route for scripts; subject to the proxy's first-byte limit when uncached."""
    try:
        return await run_in_threadpool(fn, *args)
    except GenerationError as e:
        raise HTTPException(e.status_code, e.message)


@app.post("/concept")
async def concept(req: ConceptRequest):
    if generate_contrast_set is None:
        raise HTTPException(503, "generation not available")
    return generation_stream(generate_contrast_set, (req.concept,), "concept", "examples")


@app.post("/concept.json", response_model=ConceptResponse)
async def concept_json(req: ConceptRequest):
    if generate_contrast_set is None:
        raise HTTPException(503, "generation not available")
    return await generation_json(generate_contrast_set, (req.concept,))


@app.post("/axis")
async def axis(req: AxisRequest):
    if generate_axis_set is None:
        raise HTTPException(503, "axis generation not available")
    return generation_stream(generate_axis_set, (req.concept, req.contrast_set), "axis", "axis pairs")


@app.post("/axis.json", response_model=AxisResponse)
async def axis_json(req: AxisRequest):
    if generate_axis_set is None:
        raise HTTPException(503, "axis generation not available")
    return await generation_json(generate_axis_set, (req.concept, req.contrast_set))


@app.post("/mine")
async def mine(req: MineRequest):
    if generate_decoys_from_seeds is None:
        raise HTTPException(503, "decoy mining not available")
    return generation_stream(generate_decoys_from_seeds, (req.concept, req.contrast_set, req.seeds), "mined", "decoys")


@app.post("/mine.json", response_model=MineResponse)
async def mine_json(req: MineRequest):
    if generate_decoys_from_seeds is None:
        raise HTTPException(503, "decoy mining not available")
    return await generation_json(generate_decoys_from_seeds, (req.concept, req.contrast_set, req.seeds))


@app.post("/fit", response_model=FitResponse)
async def fit(req: FitRequest):
    require_model()
    cs = req.contrast_set
    if len(cs.train_pos) < 2 or len(cs.train_neg) < 2:
        raise HTTPException(400, "need at least two positive and two negative training examples")
    require_within_fit_cap(example_count(cs), "examples")
    probe = await run_fit(fit_probe, RUNNER, req.concept, cs)
    STORE.save(probe)
    return probe.fit


@app.post("/deflate", response_model=DeflateResponse)
async def deflate(req: DeflateRequest):
    """Refit a stored probe with a confound axis projected out of every activation. The result is a new
    probe (id from the model, the contrast set and the axis set) that /chat and /score read the same way."""
    require_model()
    probe = require_probe(req.probe_id)
    pairs = req.axis_set.absent_pairs + req.axis_set.present_pairs
    if len(pairs) < 2:
        raise HTTPException(400, "need at least two axis pairs")
    require_within_fit_cap(2 * len(pairs), "axis texts (two per pair)")
    if any(not p.high.strip() or not p.low.strip() for p in pairs):
        raise HTTPException(400, "every axis pair needs a non-empty high and low text")
    cs = ContrastSet.model_validate(probe.contrast_set)
    rounds = max(1, min(MAX_DEFLATE_ROUNDS, req.rounds))
    result = await run_fit(deflate_probe, RUNNER, probe.fit["concept"], cs, req.axis_set, rounds)
    STORE.save(result.probe)
    return DeflateResponse(
        fit=result.probe.fit,
        axis=req.axis_set.axis,
        axis_cv_auroc_before=result.axis_cv_auroc_before,
        axis_cv_auroc_after=result.axis_cv_auroc_after,
        heldout_auroc_before=result.heldout_auroc_before,
        heldout_auroc_after=result.heldout_auroc_after,
        confound_index_before=result.confound_index_before,
        confound_index_after=result.confound_index_after,
        rounds_applied=result.rounds_applied,
        round_history=result.round_history,
    )


@app.post("/misfires", response_model=MisfiresResponse)
async def misfires_route(req: MisfiresRequest):
    """What a stored probe fires on when the concept is absent: the top tokens of its concept-absent training
    examples at one layer, read through the same encoding path as the fit."""
    require_model()
    probe = require_probe(req.probe_id)
    layer = probe.picked_layer if req.layer is None else req.layer
    validate_layer(layer)
    cs = ContrastSet.model_validate(probe.contrast_set)
    try:
        return await run_in_threadpool(misfires, RUNNER, probe, cs, layer, max(1, min(req.top_k, 200)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/probe/{probe_id}", response_model=ProbeInfo)
async def probe_info(probe_id: str):
    probe = STORE.get(probe_id)
    if probe is None:
        raise HTTPException(404, f"probe {probe_id!r} not found")
    return {**probe.fit, "contrast_set": probe.contrast_set}


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    require_model()
    probe = require_probe(req.probe_id)
    validate_conversation(req.system_prompt, req.messages)
    add_generation_prompt = req.messages[-1].role == "user"
    prompt_ids, spans = await run_in_threadpool(render_and_check, req.system_prompt, req.messages, add_generation_prompt)
    if len(prompt_ids) > config.MAX_CONVERSATION_TOKENS:
        raise HTTPException(400, f"conversation is over {config.MAX_CONVERSATION_TOKENS} tokens; reset to continue")

    def work() -> ScoreResponse:
        with RUNNER.lock:
            proj = RUNNER.score_ids(prompt_ids, *probe_on_device(probe))
        return ScoreResponse(turns=build_turns(prompt_ids, spans, proj, probe), conversation_tokens=len(prompt_ids))

    return await run_in_threadpool(work)


@app.post("/chat")
async def chat(req: ChatRequest):
    require_model()
    probe = require_probe(req.probe_id)
    validate_conversation(req.system_prompt, req.messages)
    if req.messages[-1].role != "user":
        raise HTTPException(400, "the last message must be from the user")
    layer = probe.picked_layer if req.layer is None else req.layer
    validate_layer(layer)
    prompt_ids, spans = await run_in_threadpool(render_and_check, req.system_prompt, req.messages, True)
    budget = config.MAX_CONVERSATION_TOKENS - len(prompt_ids)
    if budget < MIN_REPLY_TOKENS:
        raise HTTPException(400, f"conversation is at the {config.MAX_CONVERSATION_TOKENS} token limit; reset to continue")
    max_new = max(1, min(req.max_new_tokens, config.MAX_NEW_TOKENS_CAP, budget))
    alpha = max(-ALPHA_LIMIT, min(ALPHA_LIMIT, req.alpha))
    temperature = max(0.0, min(2.0, req.temperature))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            while not RUNNER.lock.acquire(timeout=KEEPALIVE_SECONDS):
                emit({"type": "token", "text": ""})
                if cancelled.is_set():
                    return
            try:
                steer = None
                if alpha != 0.0:
                    vector = alpha * probe.mean_norm[layer] * probe.dhat[layer]
                    steer = (layer, RUNNER.to_device(vector.astype(np.float32)))
                gen_ids, hit_cap = RUNNER.generate(
                    prompt_ids, max_new, temperature, req.seed, steer,
                    on_text=lambda text: emit({"type": "token", "text": text}),
                    cancelled=cancelled,
                )
                if cancelled.is_set():
                    return
                ids = prompt_ids + gen_ids
                reply = RUNNER.tok.decode(gen_ids, skip_special_tokens=True).rstrip(REPLACEMENT_CHAR)
                all_spans = spans + [Span("assistant", reply, len(prompt_ids), len(ids))]
                proj = RUNNER.score_ids(ids, *probe_on_device(probe))
            finally:
                RUNNER.lock.release()
            scored = ChatScored(
                reply=reply,
                turns=build_turns(ids, all_spans, proj, probe),
                hit_length_cap=hit_cap,
                conversation_tokens=len(ids),
                flags=flags_for(reply),
            )
            emit({"type": "scored", **scored.model_dump()})
        except Exception as e:
            log.exception("chat failed")
            emit({"type": "error", "error": f"{type(e).__name__}: {e}"})

    threading.Thread(target=work, name="chat", daemon=True).start()

    async def sse():
        try:
            while True:
                event = await queue.get()
                yield sse_line(event)
                if event["type"] in ("scored", "error"):
                    return
        finally:
            cancelled.set()

    return StreamingResponse(sse(), media_type="text/event-stream", headers=SSE_HEADERS)


# ------------------------------------------------------------------ shared conversation logic


def validate_conversation(system_prompt: str, messages) -> None:
    """Cheap checks before any tokenization: counts and character totals bound the rendering work."""
    if not messages:
        raise HTTPException(400, "messages must not be empty")
    if len(messages) > MAX_MESSAGES:
        raise HTTPException(400, f"at most {MAX_MESSAGES} messages; reset to continue")
    if len(system_prompt) > config.MAX_SYSTEM_PROMPT_CHARS:
        raise HTTPException(400, f"system prompt is over {config.MAX_SYSTEM_PROMPT_CHARS} characters")
    if len(system_prompt) + sum(len(m.content) for m in messages) > MAX_CONVERSATION_CHARS:
        raise HTTPException(400, f"conversation is over {MAX_CONVERSATION_CHARS} characters; reset to continue")
    for m in messages:
        if not m.content.strip():
            raise HTTPException(400, "messages must not be empty")
        if len(m.content) > config.MAX_MESSAGE_CHARS:
            raise HTTPException(400, f"a message is over {config.MAX_MESSAGE_CHARS} characters")


def validate_layer(layer: int) -> None:
    if not 0 <= layer < RUNNER.n_layers:
        raise HTTPException(400, f"layer must be in [0, {RUNNER.n_layers})")


def probe_on_device(probe: Probe) -> tuple:
    """(dhat, mid, axis basis or None) as device tensors, the arguments ``Runner.score_ids`` reads."""
    basis = RUNNER.to_device(probe.basis) if probe.basis is not None else None
    return RUNNER.to_device(probe.dhat), RUNNER.to_device(probe.mid), basis


def render_and_check(system_prompt: str, messages, add_generation_prompt: bool) -> tuple[list[int], list[Span]]:
    try:
        rendered = RUNNER.render_conversation(system_prompt, [(m.role, m.content) for m in messages], add_generation_prompt)
        ids = RUNNER.encode_ids(rendered.prompt)
        RUNNER.check_spans(ids, rendered.spans)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ids, rendered.spans


def build_turns(ids: list[int], spans: list[Span], proj: np.ndarray, probe: Probe) -> list[ScoredTurn]:
    """Per-turn token strings, token z [T][L] and sequence z [L] from raw projections [T_total][L]."""
    turns = []
    for s in spans:
        start = max(s.start, 1)
        p = proj[start : s.end]
        turns.append(
            ScoredTurn(
                role=s.role,
                text=s.text,
                tokens=token_strings(RUNNER.tok, ids[start : s.end]),
                z=np.round(probe.tok_z(p), 3).tolist(),
                seq_z=np.round(probe.seq_z(p.mean(axis=0)), 3).tolist() if len(p) else [0.0] * probe.n_layers,
            )
        )
    return turns
