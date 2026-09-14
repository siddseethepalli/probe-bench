import { useCallback, useEffect, useRef, useState } from "react";
import type { AxisSet, ContrastSet, FitResponse } from "./types";

// A probe the user chose to keep: everything the panels need to render it offline and
// everything the server needs to rebuild it when the pod has forgotten the id.
export interface SavedProbe {
  name: string;
  probe_id: string;
  concept: string;
  deflated_axis: string | null;
  contrast_set: ContrastSet;
  axis_set: AxisSet | null;
  axis_rounds?: number; // deflation rounds; absent on entries saved before rounds existed
  fit: FitResponse;
  system_prompt: string;
  saved_at: string;
  your_examples: string[];
  mined_examples: string[];
}

export const SAVED_CAP = 20;
const STORAGE_KEY = "probe-bench.saved.v1";

function isSavedProbe(value: unknown): value is SavedProbe {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    typeof record.name === "string" &&
    typeof record.probe_id === "string" &&
    typeof record.concept === "string" &&
    typeof record.contrast_set === "object" &&
    record.contrast_set !== null &&
    typeof record.fit === "object" &&
    record.fit !== null
  );
}

function readSaved(): SavedProbe[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isSavedProbe).map((entry) => ({
      ...entry,
      deflated_axis: typeof entry.deflated_axis === "string" ? entry.deflated_axis : null,
      axis_set: entry.axis_set && typeof entry.axis_set === "object" ? entry.axis_set : null,
      system_prompt: typeof entry.system_prompt === "string" ? entry.system_prompt : "",
      saved_at: typeof entry.saved_at === "string" ? entry.saved_at : "",
      your_examples: Array.isArray(entry.your_examples) ? entry.your_examples.filter((text): text is string => typeof text === "string") : [],
      mined_examples: Array.isArray(entry.mined_examples) ? entry.mined_examples.filter((text): text is string => typeof text === "string") : [],
    }));
  } catch {
    return [];
  }
}

// Name suggested for the current probe: the concept plus what was done to it.
export function defaultProbeName(concept: string, contrastSet: ContrastSet, fit: FitResponse, minedCount: number): string {
  const decoyTexts = new Set(contrastSet.decoys.map((example) => example.text));
  const parts: string[] = [];
  if (contrastSet.train_neg.some((example) => decoyTexts.has(example.text))) {
    parts.push("after adding decoys");
  }
  if (fit.deflated_axis) {
    parts.push(`${fit.deflated_axis} projected out`);
  }
  if (minedCount > 0) {
    parts.push("mined");
  }
  return `${concept}${parts.map((part) => ` (${part})`).join("")}`;
}

// Saved probes live under their own localStorage key, newest first, capped at SAVED_CAP
// with the oldest evicted. Saving an id that is already there replaces that entry.
export function useSavedProbes(): {
  saved: SavedProbe[];
  save: (entry: SavedProbe) => SavedProbe | null;
  remove: (probeId: string) => void;
  storageNote: string | null;
} {
  const [saved, setSaved] = useState<SavedProbe[]>(readSaved);
  const [storageNote, setStorageNote] = useState<string | null>(null);
  const savedRef = useRef(saved);
  savedRef.current = saved;

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
      setStorageNote(null);
    } catch {
      setStorageNote("Browser storage is full, so this list is not being kept; remove a saved probe or two.");
    }
  }, [saved]);

  const save = useCallback((entry: SavedProbe): SavedProbe | null => {
    const rest = savedRef.current.filter((probe) => probe.probe_id !== entry.probe_id);
    const next = [entry, ...rest];
    const evicted = next.length > SAVED_CAP ? next[next.length - 1] : null;
    const kept = evicted ? next.slice(0, SAVED_CAP) : next;
    savedRef.current = kept;
    setSaved(kept);
    return evicted;
  }, []);

  const remove = useCallback((probeId: string) => {
    const kept = savedRef.current.filter((probe) => probe.probe_id !== probeId);
    savedRef.current = kept;
    setSaved(kept);
  }, []);

  return { saved, save, remove, storageNote };
}
