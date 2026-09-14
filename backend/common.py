"""Model loading and prompt rendering. Copied from the emotion-probes tutorial with ``content_span``
generalized to locate a text inside any rendered prompt."""

from __future__ import annotations

import functools

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# Pin the exact checkpoint so activations and probes are reproducible across machines.
PINNED_REVISIONS = {
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str = DEFAULT_MODEL, revision: str | None = None, device: str | None = None):
    """Load tokenizer + model in bf16 on the best available device.

    ``AutoModelForCausalLM`` resolves a ``qwen3_5``-family checkpoint (which
    includes Qwen3.8-27B) to the text-only ``Qwen3_5ForCausalLM`` class, so the
    vision tower is never instantiated and the decoder blocks live at
    ``model.model.layers``.
    """
    revision = revision or PINNED_REVISIONS.get(model_id)
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # Right padding keeps every real token a clean causal prefix. Qwen3.5/3.8
    # mix full attention with linear (recurrent) attention blocks, and a
    # recurrent state should never have to "see" pad tokens before real ones.
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": torch.bfloat16, "revision": revision}
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return tok, model


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The stack of decoder blocks, whichever attribute path this architecture uses."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        try:
            layers = functools.reduce(getattr, path.split("."), model)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList):
            return layers
    raise AttributeError("could not locate the decoder layer stack on this model")


def render_prompt(tok: PreTrainedTokenizerBase, text: str) -> str:
    """Wrap ``text`` as a single user turn and open the assistant turn.

    ``enable_thinking=False`` matters twice over: it closes the reasoning block
    (``<think>\\n\\n</think>``) so the final token is a fixed, content-free
    position to read from, and it stops the template from injecting a
    "Reasoning effort is set to xhigh..." system line that would otherwise sit
    in front of every prompt.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def content_span(tok: PreTrainedTokenizerBase, rendered: str, text: str, search_from: int = 0) -> tuple[int, int]:
    """Token index range [start, end) of ``text`` inside ``rendered``, searching from character ``search_from``.

    Token offsets come from tokenizing the character prefix up to the text and up to its end, which is
    exact whenever the text sits between template markers (a newline before it, a special token after).
    Callers verify by decoding the span back to the text.
    """
    at = rendered.find(text, search_from)
    if at < 0:
        raise ValueError(f"text not found in the rendered prompt: {text[:60]!r}")
    start = len(tok(rendered[:at], add_special_tokens=False).input_ids)
    end = len(tok(rendered[: at + len(text)], add_special_tokens=False).input_ids)
    return start, end
