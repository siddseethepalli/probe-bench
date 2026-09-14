"""Single source of configuration. Everything reads from here; env vars override the model choice."""

import os

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
REVISION = os.environ.get("REVISION", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DTYPE = "bfloat16"

MAX_TEXT_TOKENS = 96
MAX_EXAMPLES_PER_FIT = 160
MAX_CONVERSATION_TOKENS = 6000
MAX_NEW_TOKENS_DEFAULT = 400
MAX_NEW_TOKENS_CAP = 800
MAX_SYSTEM_PROMPT_CHARS = 2000
MAX_MESSAGE_CHARS = 4000

DAILY_GENERATION_CAP = 200
CONCEPT_MAX_CHARS = 60
GEN_MODEL = "claude-opus-5"  # writes the examples (three concurrent calls per set)
GEN_EFFORT = "medium"
GEN_PLAN_MODEL = "claude-sonnet-5"  # writes the small plan (keywords, topics, prompts) that the three calls share
GEN_PLAN_EFFORT = "low"
GEN_MAX_TOKENS = 8000  # per call; the largest call writes ~42 examples with contexts, about 3.5k tokens before thinking
GEN_TIMEOUT_S = 300  # streaming, so this bounds the gap between chunks, not the whole call; heartbeats cover the wait

PROBE_WITH_TEMPLATE = True
PORT = 8000
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")  # access password for every route but /health; empty leaves the server open
DATA_DIR = os.environ.get("DATA_DIR", "data")  # probes/ and cache/ live under here
