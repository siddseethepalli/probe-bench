import type {
  AppConfig,
  AxisEvent,
  ChatEvent,
  ChatRequest,
  ConceptEvent,
  ContrastSet,
  DeflateRequest,
  DeflateResponse,
  FitResponse,
  Health,
  MineEvent,
  MineRequest,
  MisfiresRequest,
  MisfiresResponse,
  ProbeInfo,
  ScoreRequest,
  ScoreResponse,
  StaticExample,
} from "./types";
import { getStoredKey } from "./auth";

export interface ExampleSummary {
  slug: string;
  title: string;
  blurb: string;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function describeError(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  let message = `The server answered ${res.status}${res.statusText ? ` ${res.statusText}` : ""}.`;
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object") {
      const record = body as Record<string, unknown>;
      if (typeof record.error === "string") {
        message = record.error;
      } else if (typeof record.detail === "string") {
        message = record.detail;
      } else if (record.detail !== undefined) {
        message = JSON.stringify(record.detail);
      }
    }
  } catch {
    // Non-JSON error body: keep the status line.
  }
  return new ApiError(res.status, message);
}

// Called when any API route answers 401, so the page can drop the stored password and show the
// gate again.
let unauthorizedHandler: (() => void) | null = null;

export function onUnauthorized(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

// Every request carries the access password from storage as X-Probe-Key.
function withKey(init?: RequestInit): RequestInit {
  const key = getStoredKey();
  if (!key) {
    return init ?? {};
  }
  const headers = new Headers(init?.headers);
  headers.set("X-Probe-Key", key);
  return { ...init, headers };
}

async function send(url: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, withKey(init));
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    throw new ApiError(0, `Could not reach the server (${describeError(err)}).`);
  }
  if (res.status === 401) {
    unauthorizedHandler?.();
  }
  if (!res.ok) {
    throw await errorFromResponse(res);
  }
  return res;
}

// GET /auth checks a candidate password without storing it: true on 200, false on 401.
export async function checkKey(apiBase: string, key: string): Promise<boolean> {
  let res: Response;
  try {
    res = await fetch(`${apiBase}/auth`, { headers: { "X-Probe-Key": key }, cache: "no-store" });
  } catch (err) {
    throw new ApiError(0, `Could not reach the server (${describeError(err)}).`);
  }
  if (res.status === 401) {
    return false;
  }
  if (!res.ok) {
    throw await errorFromResponse(res);
  }
  return true;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await send(url, init);
  return (await res.json()) as T;
}

function jsonPost(body: unknown, signal?: AbortSignal): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  };
}

export function fetchConfig(): Promise<AppConfig> {
  return requestJson<AppConfig>("/config.json", { cache: "no-store" });
}

export function fetchExampleIndex(): Promise<ExampleSummary[]> {
  return requestJson<ExampleSummary[]>("/examples/index.json");
}

export function fetchExample(slug: string): Promise<StaticExample> {
  return requestJson<StaticExample>(`/examples/${encodeURIComponent(slug)}.json`);
}

export async function getHealth(apiBase: string, timeoutMs: number): Promise<Health> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await requestJson<Health>(`${apiBase}/health`, { signal: controller.signal, cache: "no-store" });
  } finally {
    window.clearTimeout(timer);
  }
}

export function postFit(apiBase: string, concept: string, contrastSet: ContrastSet): Promise<FitResponse> {
  return requestJson<FitResponse>(`${apiBase}/fit`, jsonPost({ concept, contrast_set: contrastSet }));
}

export function postScore(apiBase: string, req: ScoreRequest): Promise<ScoreResponse> {
  return requestJson<ScoreResponse>(`${apiBase}/score`, jsonPost(req));
}

export function getProbe(apiBase: string, probeId: string): Promise<ProbeInfo> {
  return requestJson<ProbeInfo>(`${apiBase}/probe/${encodeURIComponent(probeId)}`);
}

export function postDeflate(apiBase: string, req: DeflateRequest): Promise<DeflateResponse> {
  return requestJson<DeflateResponse>(`${apiBase}/deflate`, jsonPost(req));
}

export function postMisfires(apiBase: string, req: MisfiresRequest): Promise<MisfiresResponse> {
  return requestJson<MisfiresResponse>(`${apiBase}/misfires`, jsonPost(req));
}

function parseEventBlock<T>(block: string): T | null {
  const data = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (data.trim() === "") {
    return null;
  }
  return JSON.parse(data) as T;
}

// POST routes that answer with server-sent events. EventSource cannot POST, so the
// body is read as a stream and split on blank lines by hand.
async function streamEvents<T>(url: string, body: unknown, onEvent: (event: T) => void, signal?: AbortSignal): Promise<void> {
  const res = await send(url, { ...jsonPost(body, signal), headers: { "Content-Type": "application/json", Accept: "text/event-stream" } });
  if (!res.body) {
    throw new ApiError(0, "The server sent no reply body.");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const drain = (flush: boolean) => {
    buffer = buffer.replace(/\r\n/g, "\n");
    let cut = buffer.indexOf("\n\n");
    while (cut !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = parseEventBlock<T>(block);
      if (event) {
        onEvent(event);
      }
      cut = buffer.indexOf("\n\n");
    }
    if (flush && buffer.trim() !== "") {
      const event = parseEventBlock<T>(buffer);
      buffer = "";
      if (event) {
        onEvent(event);
      }
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    drain(false);
  }
  buffer += decoder.decode();
  drain(true);
}

export function streamChat(apiBase: string, req: ChatRequest, onEvent: (event: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  return streamEvents<ChatEvent>(`${apiBase}/chat`, req, onEvent, signal);
}

export function streamConcept(apiBase: string, concept: string, onEvent: (event: ConceptEvent) => void, signal?: AbortSignal): Promise<void> {
  return streamEvents<ConceptEvent>(`${apiBase}/concept`, { concept }, onEvent, signal);
}

// The axis to fit travels as contrast_set.confound_axis; the request has no field of its own.
export function streamAxis(
  apiBase: string,
  concept: string,
  contrastSet: ContrastSet,
  onEvent: (event: AxisEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamEvents<AxisEvent>(`${apiBase}/axis`, { concept, contrast_set: contrastSet }, onEvent, signal);
}

export function streamMine(apiBase: string, req: MineRequest, onEvent: (event: MineEvent) => void, signal?: AbortSignal): Promise<void> {
  return streamEvents<MineEvent>(`${apiBase}/mine`, req, onEvent, signal);
}
