"""Everything that touches the model: loading, rendering examples and conversations, span location,
pooled and per-token residual readouts, generation with the steering hook.

Layer index ``l`` in [0, n_layers) means ``hidden_states[l + 1]``, the output of decoder block ``l``
before the final norm. Position 0 (the attention sink) is never included in any span.
All GPU work holds ``Runner.lock``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

from backend import config
from backend.common import content_span, decoder_layers, load_model, render_prompt

log = logging.getLogger("probe-bench")

TURN_HEADER = "<|im_start|>{role}\n"
MAX_CONTEXT_TOKENS = 4 * config.MAX_TEXT_TOKENS  # a context is read for its effect on the span, not pooled, so it may run longer
REPLACEMENT_CHAR = "�"  # what decoding an incomplete UTF-8 sequence yields


def token_strings(tok, ids: list[int]) -> list[str]:
    """One display string per token id, by incremental decoding. Byte-level BPE splits a multibyte character
    (emoji, CJK) across tokens and a partial character decodes to U+FFFD, so each token shows only its real
    text with any partial-byte tail dropped (often "") and the completing token carries the whole character.
    The window restarts at every clean character boundary, so the cost stays linear in the number of tokens."""
    out: list[str] = []
    anchor, previous = 0, ""
    for i in range(len(ids)):
        cumulative = tok.decode(ids[anchor : i + 1])
        base = previous.rstrip(REPLACEMENT_CHAR)
        piece = cumulative[len(base) :] if cumulative.startswith(base) else cumulative
        out.append(piece.rstrip(REPLACEMENT_CHAR))
        if cumulative.endswith(REPLACEMENT_CHAR):
            previous = cumulative
        else:
            anchor, previous = i + 1, ""
    return out


@dataclass
class Span:
    """A turn or example located inside a rendered prompt: token range [start, end) plus its text."""

    role: str
    text: str
    start: int
    end: int


@dataclass
class Rendered:
    prompt: str
    spans: list[Span]


class Runner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tok = None
        self.model = None
        self.loading = False
        self.n_layers = 0
        self.d_model = 0
        self.device = None
        self._eos_ids: set[int] = set()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.loading = True
        t0 = time.time()
        try:
            revision = config.REVISION if config.MODEL_ID != config.FALLBACK_MODEL_ID else None
            log.info("loading %s (revision %s)", config.MODEL_ID, revision)
            # On sm90 GPUs PyTorch's default attention backend order picks cuDNN SDPA, which rebuilds an
            # execution plan on the CPU for every new KV length, and decode presents a new length every
            # step (2.6 tok/s on an H100 with it, 20 without). Harmless on sm80, which is never offered it.
            torch.backends.cuda.enable_cudnn_sdp(False)
            tok, model = load_model(config.MODEL_ID, revision)
            self.n_layers = len(decoder_layers(model))
            self.d_model = int(model.config.hidden_size)
            self.device = next(model.parameters()).device
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            self._eos_ids = {int(i) for i in eos if i is not None} | {int(tok.eos_token_id), int(tok.pad_token_id)}
            self.tok, self.model = tok, model
            log.info("loaded %s: %d layers, d=%d, in %.0fs", config.MODEL_ID, self.n_layers, self.d_model, time.time() - t0)
        except Exception:
            log.exception("model load failed")
        finally:
            self.loading = False

    # ------------------------------------------------------------------ rendering

    def render_example(self, text: str, context: str | None) -> Rendered:
        """An example without context is a single user turn; with context it is the assistant reply to
        that user turn. Thinking is off in both, so the prefix is fixed and content-free."""
        text = text.strip()
        if not text:
            raise ValueError("an example has empty text")
        context = context.strip() if context and context.strip() else None
        if context:
            if len(self.tok(context, add_special_tokens=False).input_ids) > MAX_CONTEXT_TOKENS:
                raise ValueError(f"context longer than {MAX_CONTEXT_TOKENS} tokens: {context[:60]!r}")
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": context}, {"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            search_from = prompt.find(context) + len(context)
        else:
            prompt = render_prompt(self.tok, text)
            search_from = 0
        start, end = content_span(self.tok, prompt, text, search_from)
        if end - start > config.MAX_TEXT_TOKENS:
            raise ValueError(f"example longer than {config.MAX_TEXT_TOKENS} tokens: {text[:60]!r}")
        return Rendered(prompt, [Span("assistant" if context else "user", text, max(start, 1), end)])

    def render_conversation(self, system_prompt: str, messages: list[tuple[str, str]], add_generation_prompt: bool) -> Rendered:
        """Render system prompt plus messages and locate every turn in order with a moving cursor."""
        turns = [("system", system_prompt.strip())] if system_prompt.strip() else []
        turns += [(role, content.strip()) for role, content in messages]
        prompt = self.tok.apply_chat_template(
            [{"role": r, "content": c} for r, c in turns],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        spans, cursor = [], 0
        for role, content in turns:
            header_at = prompt.find(TURN_HEADER.format(role=role), cursor)
            search_from = header_at + len(TURN_HEADER.format(role=role)) if header_at >= 0 else cursor
            start, end = content_span(self.tok, prompt, content, search_from)
            spans.append(Span(role, content, max(start, 1), end))
            cursor = prompt.find(content, search_from) + len(content)
        return Rendered(prompt, spans)

    def encode_ids(self, prompt: str) -> list[int]:
        return self.tok(prompt, add_special_tokens=False).input_ids

    def check_spans(self, ids: list[int], spans: list[Span]) -> None:
        for s in spans:
            got = self.tok.decode(ids[s.start : s.end])
            if got != s.text:
                raise ValueError(f"token span does not decode back to the text: {got[:60]!r} vs {s.text[:60]!r}")

    # ------------------------------------------------------------------ readouts

    @torch.no_grad()
    def encode_examples(self, items: list[Rendered], keep_tokens: int = 0, batch_size: int = 16) -> tuple[np.ndarray, list[torch.Tensor | None]]:
        """Pooled residuals [N, L, d] float32 (mean over each example's span), plus, for the first
        ``keep_tokens`` items, their per-token residuals as GPU tensors [T, L, d] in the model dtype."""
        order = sorted(range(len(items)), key=lambda i: len(items[i].prompt))
        pooled = np.zeros((len(items), self.n_layers, self.d_model), dtype=np.float32)
        tokens: list[torch.Tensor | None] = [None] * len(items)
        with self.lock:
            for i0 in range(0, len(order), batch_size):
                chunk = order[i0 : i0 + batch_size]
                enc = self.tok([items[i].prompt for i in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
                for b, i in enumerate(chunk):
                    self.check_spans(enc.input_ids[b].tolist(), items[i].spans)
                out = self.model(**enc.to(self.device), output_hidden_states=True)
                hs = out.hidden_states[1:]
                assert len(hs) == self.n_layers, f"expected {self.n_layers} hidden states, got {len(hs)}"
                for b, i in enumerate(chunk):
                    span = items[i].spans[0]
                    reps = torch.stack([h[b, span.start : span.end] for h in hs], dim=1)  # [T, L, d]
                    pooled[i] = reps.float().mean(dim=0).cpu().numpy()
                    if i < keep_tokens:
                        tokens[i] = reps
                del out, hs
        return pooled, tokens

    @torch.no_grad()
    def project_tokens(self, reps_list: list[torch.Tensor], dhat: np.ndarray, mid: np.ndarray, basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Raw projections ``x . dhat_l - mid_l`` [Ntok, L] and residual norms [Ntok, L] over the given token reps.
        With ``basis`` ([k, L, d] unit rows) every rep is deflated first, one axis at a time."""
        dhat_t = torch.from_numpy(dhat).to(self.device)
        mid_t = torch.from_numpy(mid).to(self.device)
        basis_t = torch.from_numpy(basis).to(self.device) if basis is not None else None
        projs, norms = [], []
        with self.lock:
            for reps in reps_list:
                x = reps.float()
                if basis_t is not None:
                    for axis_t in basis_t:
                        x = x - torch.einsum("tld,ld->tl", x, axis_t)[..., None] * axis_t
                projs.append((torch.einsum("tld,ld->tl", x, dhat_t) - mid_t).cpu().numpy())
                norms.append(x.norm(dim=-1).cpu().numpy())
        return np.concatenate(projs), np.concatenate(norms)

    @torch.no_grad()
    def score_ids(self, ids: list[int], dhat: torch.Tensor, mid: torch.Tensor, basis: torch.Tensor | None = None) -> np.ndarray:
        """One unsteered forward over exactly ``ids``; raw projection of every token at every layer [T, L],
        after deflating each token rep along every axis of ``basis`` [k, L, d] when given. Caller holds the lock."""
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
        cols = []
        for l, h in enumerate(out.hidden_states[1:]):
            r = h[0].float()
            if basis is not None:
                for axis in basis[:, l]:
                    r = r - (r @ axis)[:, None] * axis
            cols.append(r @ dhat[l] - mid[l])
        proj = torch.stack(cols, dim=1).cpu().numpy()
        del out
        return proj

    def to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        seed: int,
        steer: tuple[int, torch.Tensor] | None,
        on_text: Callable[[str], None],
        cancelled: threading.Event,
    ) -> tuple[list[int], bool]:
        """Stream a reply for ``prompt_ids``; returns (generated ids without EOS, hit_length_cap).
        ``steer`` is (decoder block index, vector added at every position). Caller holds the lock."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x = torch.tensor([prompt_ids], device=self.device)
        kwargs = dict(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            streamer=_CallbackStreamer(self.tok, on_text),
            stopping_criteria=StoppingCriteriaList([_Cancelled(cancelled)]),
        )
        if temperature > 0.01:
            kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        steer_ctx = add_direction(self.model, steer[0], steer[1]) if steer is not None else contextlib.nullcontext()
        with steer_ctx:
            out = self.model.generate(**kwargs)
        raw = out[0, len(prompt_ids) :].tolist()
        gen = list(raw)
        while gen and gen[-1] in self._eos_ids:
            gen.pop()
        hit_cap = len(raw) >= max_new_tokens and (not raw or raw[-1] not in self._eos_ids)
        return gen, hit_cap


class _CallbackStreamer(TextStreamer):
    """TextStreamer that hands each finalized text chunk to a callback instead of printing."""

    def __init__(self, tok, on_text: Callable[[str], None]) -> None:
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self._on_text = on_text

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if stream_end:
            text = text.rstrip(REPLACEMENT_CHAR)  # a length cap can cut a multibyte character in half
        if text:
            self._on_text(text)


class _Cancelled(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


@contextlib.contextmanager
def add_direction(model, block: int, direction: torch.Tensor):
    """Add ``direction`` to the output of decoder block ``block`` at every position (prefill and decode).
    Probe layer ``l`` reads ``hidden_states[l + 1]``, the output of block ``l``, so it hooks block ``l``."""
    layer = decoder_layers(model)[block]

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + direction.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def flags_for(text: str) -> list[str]:
    """Degeneration and template-leak detectors, from the emotion-steering playground."""
    from collections import Counter

    f = set()
    if not text.strip():
        return ["empty"]
    toks = text.split()
    if len(toks) >= 12:
        bg = list(zip(toks, toks[1:]))
        if len(set(bg)) / max(1, len(bg)) < 0.35:
            f.add("degenerate")
        tg = list(zip(toks, toks[1:], toks[2:]))
        if tg and Counter(tg).most_common(1)[0][1] >= 4:
            f.add("degenerate")
    if any(m in text for m in ("<|im_start|>", "<|im_end|>", "</think>", "<think>")):
        f.add("template_leak")
    return sorted(f)
