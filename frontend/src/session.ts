import { useCallback, useEffect, useState } from "react";
import type { AxisSet, ContrastSet, Message } from "./types";

// Everything the page needs to rebuild itself after a reload, as text only.
// Activations never go here: the server recolors the conversation on demand.
export interface Session {
  concept: string;
  contrastSet: ContrastSet | null;
  probeId: string | null;
  systemPrompt: string;
  messages: Message[];
  selectedLayer: number | null;
  alpha: number;
  maxNewTokens: number;
  exampleSlug: string | null;
  axisSet: AxisSet | null; // confound-axis pairs for the current concept, reused across deflations
  yourExamples: string[]; // texts typed into the training set by hand, for the "yours" tag
  dirty: boolean; // the training set differs from the one the current probe was fit on
  minedExamples: string[]; // decoy texts written around misfires and added as negatives, for the "mined" tag
  axisRounds: number; // deflation rounds behind the fit on screen, replayed when the probe is rebuilt
}

export const DEFAULT_MAX_NEW_TOKENS = 400;
export const ALPHA_LIMIT = 1;

export function clampAlpha(value: number): number {
  return Math.max(-ALPHA_LIMIT, Math.min(ALPHA_LIMIT, value));
}

export const ROUNDS_MIN = 1;
export const ROUNDS_MAX = 4;

export function clampRounds(value: unknown): number {
  const rounds = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : ROUNDS_MIN;
  return Math.max(ROUNDS_MIN, Math.min(ROUNDS_MAX, rounds));
}

export const EMPTY_SESSION: Session = {
  concept: "",
  contrastSet: null,
  probeId: null,
  systemPrompt: "",
  messages: [],
  selectedLayer: null,
  alpha: 0,
  maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
  exampleSlug: null,
  axisSet: null,
  yourExamples: [],
  dirty: false,
  minedExamples: [],
  axisRounds: 1,
};

const STORAGE_KEY = "probe-bench.session.v2";

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (record.role === "user" || record.role === "assistant") && typeof record.content === "string";
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readStoredSession(): Session {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return EMPTY_SESSION;
    }
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return EMPTY_SESSION;
    }
    const record = parsed as Record<string, unknown>;
    const contrastSet = record.contrastSet && typeof record.contrastSet === "object" ? (record.contrastSet as ContrastSet) : null;
    const axisSet = record.axisSet && typeof record.axisSet === "object" ? (record.axisSet as AxisSet) : null;
    const selectedLayer = typeof record.selectedLayer === "number" && Number.isFinite(record.selectedLayer) ? record.selectedLayer : null;
    return {
      concept: stringOr(record.concept, ""),
      contrastSet,
      probeId: nullableString(record.probeId),
      systemPrompt: stringOr(record.systemPrompt, ""),
      messages: Array.isArray(record.messages) ? record.messages.filter(isMessage) : [],
      selectedLayer,
      alpha: clampAlpha(numberOr(record.alpha, 0)),
      maxNewTokens: numberOr(record.maxNewTokens, DEFAULT_MAX_NEW_TOKENS),
      exampleSlug: nullableString(record.exampleSlug),
      axisSet,
      yourExamples: Array.isArray(record.yourExamples) ? record.yourExamples.filter((text): text is string => typeof text === "string") : [],
      dirty: record.dirty === true,
      minedExamples: Array.isArray(record.minedExamples) ? record.minedExamples.filter((text): text is string => typeof text === "string") : [],
      axisRounds: clampRounds(record.axisRounds),
    };
  } catch {
    return EMPTY_SESSION;
  }
}

export type SessionPatch = Partial<Session> | ((prev: Session) => Session);

export function useSessionState(): {
  session: Session;
  update: (patch: SessionPatch) => void;
  reset: () => void;
} {
  const [session, setSession] = useState<Session>(readStoredSession);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    } catch {
      // Storage unavailable or full: the session simply lives in memory.
    }
  }, [session]);

  const update = useCallback((patch: SessionPatch) => {
    setSession((prev) => (typeof patch === "function" ? patch(prev) : { ...prev, ...patch }));
  }, []);

  const reset = useCallback(() => {
    setSession(EMPTY_SESSION);
  }, []);

  return { session, update, reset };
}
