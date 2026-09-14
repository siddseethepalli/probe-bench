import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  checkKey,
  describeError,
  onUnauthorized,
  fetchConfig,
  fetchExample,
  fetchExampleIndex,
  getProbe,
  postDeflate,
  postFit,
  postMisfires,
  postScore,
  streamAxis,
  streamMine,
  streamChat,
  streamConcept,
  type ExampleSummary,
} from "./api";
import { ChatPanel, type ChatStatus, type CompareCandidate, type Comparison, type Ghost } from "./components/ChatPanel";
import { ConceptInput } from "./components/ConceptInput";
import { ConfoundPanel, type DecoysState, type FixEvidence } from "./components/ConfoundPanel";
import { ContrastSetPanel, DECOYS_FOR_FIX, decoysToAdd, type TrainList } from "./components/ContrastSetPanel";
import { bestConfoundLayer, describeLayer, LayerStrip } from "./components/LayerStrip";
import { useHealth } from "./health";
import { clearStoredKey, getStoredKey, setStoredKey } from "./auth";
import { Gate } from "./components/Gate";
import { InstrumentBar } from "./components/InstrumentBar";
import { SaveProbeControls } from "./components/SaveProbe";
import { Tour } from "./components/Tour";
import { defaultProbeName, useSavedProbes, type SavedProbe } from "./saved";
import { clampRounds, useSessionState } from "./session";
import { hasSeenTour, markTourSeen } from "./tour";
import type {
  AxisResponse,
  AxisSet,
  ChatRequest,
  ChatScored,
  ConceptResponse,
  ContrastSet,
  Example,
  FitResponse,
  Message,
  MineResponse,
  MisfiresResponse,
  ScoredTurn,
  StaticExample,
  DeflateRequest,
  DeflateResponse,
} from "./types";

type Busy = "load" | "concept" | "fit" | "score" | "chat" | "axis" | "deflate" | "misfires" | "mine" | null;

interface Errors {
  concept?: string;
  set?: string;
  chat?: string;
  load?: string;
  axis?: string;
  mine?: string;
}

const DEFAULT_AXIS = "warmth";

const LIST_LABELS: Record<string, string> = {
  train_pos: "positives",
  train_neg: "negatives",
  heldout_pos: "held-out positives",
  heldout_neg: "held-out negatives",
  implicit_pos: "implicit positives",
  decoys: "decoys",
  neutral: "neutral examples",
};

const MINE_LABELS: Record<string, string> = { decoys: "mined decoys" };

const AXIS_LABELS: Record<string, string> = {
  absent_pairs: "pairs with the concept absent",
  present_pairs: "pairs with the concept present",
};

function droppedNotes(dropped: Record<string, number>, lists: Record<string, unknown>, labels: Record<string, string>): string[] {
  const notes: string[] = [];
  for (const [key, count] of Object.entries(dropped)) {
    if (count <= 0) {
      continue;
    }
    const label = labels[key] ?? key.replace(/_/g, " ");
    const list = lists[key];
    if (Array.isArray(list)) {
      notes.push(`${list.length} of ${list.length + count} ${label} kept`);
    } else {
      notes.push(`${count} ${label} dropped`);
    }
  }
  return notes;
}

function messagesFromTurns(turns: ScoredTurn[]): Message[] {
  const out: Message[] = [];
  for (const turn of turns) {
    if (turn.role === "user" || turn.role === "assistant") {
      out.push({ role: turn.role, content: turn.text });
    }
  }
  return out;
}

function sameMessages(a: Message[], b: Message[]): boolean {
  return a.length === b.length && a.every((message, i) => message.role === b[i].role && message.content === b[i].content);
}

function findCanned(loaded: StaticExample, systemPrompt: string, messages: Message[]): number {
  return loaded.conversations.findIndex(
    (conversation) => conversation.system_prompt === systemPrompt && sameMessages(messagesFromTurns(conversation.scored.turns), messages),
  );
}

function clampLayer(layer: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(count - 1, Math.round(layer)));
}

function setUrl(params: Record<string, string> | null) {
  const url = new URL(window.location.href);
  url.search = "";
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, value);
    }
  }
  window.history.replaceState(null, "", url);
}

function chatStatusOf(scored: ChatScored): ChatStatus {
  return { hitLengthCap: scored.hit_length_cap, flags: scored.flags };
}

// Rounds ride along only above the default of one, so a server without the field still answers.
function deflateRequest(probeId: string, axisSet: AxisSet, rounds: number): DeflateRequest {
  const clamped = clampRounds(rounds);
  return clamped > 1 ? { probe_id: probeId, axis_set: axisSet, rounds: clamped } : { probe_id: probeId, axis_set: axisSet };
}

// The axis evidence line and its per-round table, comparing the plain fit with the deflated one.
function axisEvidence(result: DeflateResponse, plain: FitResponse, requested: number): FixEvidence {
  const applied = result.rounds_applied ?? 1;
  return {
    kind: "axis",
    label: applied > 1 ? `after projecting ${result.axis} out over ${applied} rounds` : `after projecting ${result.axis} out`,
    before: plain.confound,
    after: result.fit.confound,
    axis: {
      name: result.axis,
      before: result.axis_cv_auroc_before,
      after: result.axis_cv_auroc_after,
      heldoutBefore: result.heldout_auroc_before,
      history: result.round_history ?? [],
      requested: clampRounds(requested),
      applied,
    },
  };
}

// A probe the current conversation can be scored under, beside the one on screen.
interface CompareTarget extends CompareCandidate {
  name: string; // the probe's own name, without the "previous:" prefix the menu adds
  kind: "example" | "saved" | "previous";
  slug?: string;
  probeId?: string;
}

export default function App() {
  const { session, update, reset } = useSessionState();
  const sessionRef = useRef(session);
  sessionRef.current = session;

  const [apiBase, setApiBase] = useState<string | null>(null);
  const apiBaseRef = useRef("");
  const [configReady, setConfigReady] = useState(false);
  // The gate opens with a stored password, or without one while the server is offline.
  const [unlocked, setUnlocked] = useState(() => getStoredKey() !== null);
  const [gateNote, setGateNote] = useState<string | null>(null);
  const { status, health, retryIn, checkNow } = useHealth(apiBase);
  const online = status === "online";

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [example, setExample] = useState<StaticExample | null>(null);
  const [cannedIndex, setCannedIndex] = useState<number | null>(null);
  const [fit, setFit] = useState<FitResponse | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [turns, setTurns] = useState<ScoredTurn[]>([]);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [conceptStatus, setConceptStatus] = useState<string | null>(null);
  const [axisStatus, setAxisStatus] = useState<string | null>(null);
  const [axisNotes, setAxisNotes] = useState<string[]>([]);
  // The fit that was deflated, kept in memory so the deflation can be undone.
  const [previousFit, setPreviousFit] = useState<FitResponse | null>(null);
  const [evidence, setEvidence] = useState<FixEvidence[]>([]);
  // Scores of the same conversation under the probe shown before the last switch.
  const [ghost, setGhost] = useState<Ghost | null>(null);
  // The same conversation under a second probe; memory only, gone with the conversation.
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [misfires, setMisfires] = useState<MisfiresResponse | null>(null);
  const { saved, save, remove: removeSaved, storageNote } = useSavedProbes();
  const savedRef = useRef(saved);
  savedRef.current = saved;
  const [saveNote, setSaveNote] = useState<string | null>(null);
  const [mineStatus, setMineStatus] = useState<string | null>(null);
  const [mineNotes, setMineNotes] = useState<string[]>([]);
  const [busy, setBusy] = useState<Busy>(null);
  const [errors, setErrors] = useState<Errors>({});
  const [missingProbe, setMissingProbe] = useState<string | null>(null);
  const [tourOpen, setTourOpen] = useState(false);
  // A first visit stays armed for the tour until the page has a fit to walk through.
  const tourArmed = useRef(!hasSeenTour());
  const examplesRef = useRef<ExampleSummary[]>([]);

  // Probe id confirmed to exist on the server, so chat can skip the lookup.
  const liveProbeRef = useRef<string | null>(null);
  const chatAbortRef = useRef<AbortController | null>(null);
  const conceptAbortRef = useRef<AbortController | null>(null);
  const axisAbortRef = useRef<AbortController | null>(null);
  const mineAbortRef = useRef<AbortController | null>(null);
  const fitRef = useRef(fit);
  fitRef.current = fit;
  const exampleRef = useRef(example);
  exampleRef.current = example;
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const statusRef = useRef(status);
  statusRef.current = status;
  const ownExample = example !== null && fit !== null && fit.probe_id === example.fit.probe_id;
  // How the ghosts, the comparison and the sub-rows name the probe on screen.
  const probeLabel = ownExample
    ? example.title
    : `${session.concept || "this probe"}${fit?.deflated_axis ? `, deflated: ${fit.deflated_axis}` : ""}${
        session.minedExamples.length > 0 ? `, mined: ${session.minedExamples.length}` : ""
      }`;
  const labelRef = useRef(probeLabel);
  labelRef.current = probeLabel;

  const layerCount = fit ? fit.n_layers : 0;
  const layer = fit ? clampLayer(session.selectedLayer ?? bestConfoundLayer(fit), layerCount) : 0;
  const layerRef = useRef(layer);
  layerRef.current = layer;

  const setError = useCallback((key: keyof Errors, message: string | undefined) => {
    setErrors((prev) => ({ ...prev, [key]: message }));
  }, []);

  const stopStreaming = useCallback(() => {
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setStreamingText(null);
  }, []);

  const stopConcept = useCallback(() => {
    conceptAbortRef.current?.abort();
    conceptAbortRef.current = null;
    setConceptStatus(null);
  }, []);

  const stopAxis = useCallback(() => {
    axisAbortRef.current?.abort();
    axisAbortRef.current = null;
    setAxisStatus(null);
  }, []);

  const stopMine = useCallback(() => {
    mineAbortRef.current?.abort();
    mineAbortRef.current = null;
    setMineStatus(null);
  }, []);

  const clearDeflation = useCallback(() => {
    setPreviousFit(null);
    setEvidence((prev) => prev.filter((item) => item.kind !== "axis"));
    setAxisNotes([]);
  }, []);

  const recordEvidence = useCallback((item: FixEvidence) => {
    setEvidence((prev) => [...prev.filter((existing) => existing.kind !== item.kind), item]);
  }, []);

  // Streams the axis pairs for a concept, surfacing the server's status text when asked.
  const loadAxisSet = useCallback(
    async (base: string, concept: string, contrastSet: ContrastSet, axis: string, onStatus?: (text: string) => void, signal?: AbortSignal) => {
      let generated = null as AxisResponse | null;
      await streamAxis(
        base,
        concept || contrastSet.concept,
        { ...contrastSet, confound_axis: axis },
        (event) => {
          if (event.type === "status") {
            onStatus?.(event.text);
          } else if (event.type === "axis") {
            generated = event;
          } else {
            throw new ApiError(0, event.error);
          }
        },
        signal,
      );
      if (!generated) {
        throw new ApiError(0, "The server ended the stream without the axis pairs.");
      }
      return generated;
    },
    [],
  );

  const applyFit = useCallback(
    (fitted: FitResponse, keepLayer: boolean) => {
      setFit(fitted);
      liveProbeRef.current = fitted.probe_id;
      setMissingProbe(null);
      setMisfires(null);
      if (!fitted.deflated_axis) {
        clearDeflation();
      }
      update((prev) => ({ ...prev, probeId: fitted.probe_id, dirty: false, selectedLayer: keepLayer ? prev.selectedLayer : null }));
    },
    [clearDeflation, update],
  );

  // Projects the axis the screen shows out of a fresh fit, reusing the stored pairs when the
  // axis name matches (regenerating them otherwise) and the same number of rounds.
  const carryDeflation = useCallback(
    async (base: string, fitted: FitResponse, axis: string, contrastSet: ContrastSet): Promise<DeflateResponse> => {
      const current = sessionRef.current;
      let axisSet: AxisSet | null = current.axisSet && current.axisSet.axis === axis ? current.axisSet : null;
      if (!axisSet) {
        const generated = await loadAxisSet(base, current.concept, contrastSet, axis);
        axisSet = generated.axis_set;
        update({ axisSet });
      }
      return postDeflate(base, deflateRequest(fitted.probe_id, axisSet, current.axisRounds));
    },
    [loadAxisSet, update],
  );

  // Returns a probe id that exists on the server, refitting from the stored contrast
  // set when the saved id is gone (probes live on the pod and vanish with it). Callers
  // that changed the probe in the same tick pass the id explicitly, since the session
  // ref only catches up on the next render.
  const ensureLiveProbe = useCallback(
    async (base: string, wanted?: string): Promise<string> => {
      const current = sessionRef.current;
      const probeId = wanted ?? current.probeId;
      if (probeId && liveProbeRef.current === probeId) {
        return probeId;
      }
      if (probeId) {
        try {
          const info = await getProbe(base, probeId);
          liveProbeRef.current = info.probe_id;
          setFit(info);
          return info.probe_id;
        } catch (err) {
          if (!(err instanceof ApiError && err.status === 404)) {
            throw err;
          }
        }
      }
      if (!current.contrastSet) {
        throw new ApiError(0, "There is no contrast set to fit. Build a probe or pick a prebuilt example first.");
      }
      const fitted = await postFit(base, current.concept || current.contrastSet.concept, current.contrastSet);
      const axis = fitRef.current?.deflated_axis ?? null;
      if (!axis) {
        applyFit(fitted, true);
        return fitted.probe_id;
      }
      // The fit on screen had an axis projected out; a plain refit would silently drop that.
      const deflated = await carryDeflation(base, fitted, axis, current.contrastSet);
      applyFit(deflated.fit, true);
      return deflated.fit.probe_id;
    },
    [applyFit, carryDeflation],
  );

  const captureGhost = useCallback((label?: string, kind: "probe" | "edit" = "probe"): Ghost | null => {
    const seqZ = turnsRef.current.map((turn) => turn.seq_z);
    return seqZ.length > 0 ? { label: label ?? labelRef.current, seqZ, kind, probeId: fitRef.current?.probe_id ?? null } : null;
  }, []);

  // Colors the conversation under the probe now shown: the example's own canned turns when the
  // text matches, otherwise /score on the server, otherwise plain text until the server is back.
  const colorConversation = useCallback(
    async (
      base: string,
      probeId: string | null,
      loaded: StaticExample | null,
      ghostCandidate: Ghost | null,
      conversation?: { systemPrompt: string; messages: Message[] },
    ) => {
      const current = conversation ?? { systemPrompt: sessionRef.current.systemPrompt, messages: sessionRef.current.messages };
      if (current.messages.length === 0) {
        setTurns([]);
        setChatStatus(null);
        setGhost(null);
        return;
      }
      if (loaded && probeId === loaded.fit.probe_id) {
        const index = findCanned(loaded, current.systemPrompt, current.messages);
        if (index >= 0) {
          const canned = loaded.conversations[index];
          setCannedIndex(index);
          setTurns(canned.scored.turns);
          setChatStatus(chatStatusOf(canned.scored));
          setGhost(ghostCandidate);
          return;
        }
      }
      if (!base || statusRef.current === "offline") {
        setTurns([]);
        setChatStatus(null);
        setGhost(ghostCandidate);
        return;
      }
      setBusy("score");
      try {
        const id = probeId !== null && liveProbeRef.current === probeId ? probeId : await ensureLiveProbe(base, probeId ?? undefined);
        const scored = await postScore(base, { probe_id: id, system_prompt: current.systemPrompt, messages: current.messages });
        setTurns(scored.turns);
        setChatStatus(null);
        setGhost(ghostCandidate);
      } catch (err) {
        setTurns([]);
        setChatStatus(null);
        setGhost(ghostCandidate);
        if (err instanceof ApiError && err.status !== 0) {
          setError("chat", `Could not color the conversation: ${describeError(err)}`);
        }
      } finally {
        setBusy(null);
      }
    },
    [ensureLiveProbe, setError],
  );

  const showConversation = useCallback(
    (loaded: StaticExample, index: number) => {
      const conversation = loaded.conversations[index];
      if (!conversation) {
        setCannedIndex(null);
        setTurns([]);
        setChatStatus(null);
        update({ messages: [], systemPrompt: loaded.contrast_set.system_prompts[0] ?? "" });
        return;
      }
      setCannedIndex(index);
      setTurns(conversation.scored.turns);
      setChatStatus(chatStatusOf(conversation.scored));
      update({ systemPrompt: conversation.system_prompt, messages: messagesFromTurns(conversation.scored.turns) });
    },
    [update],
  );

  const loadExample = useCallback(
    async (slug: string) => {
      stopConcept();
      stopAxis();
      const ghostCandidate = captureGhost();
      setBusy("load");
      setErrors({});
      setMissingProbe(null);
      try {
        const loaded = await fetchExample(slug);
        stopStreaming();
        setExample(loaded);
        setFit(loaded.fit);
        clearDeflation();
        setMisfires(null);
        setMineNotes([]);
        setEvidence([]);
        setNotes([]);
        liveProbeRef.current = null;
        update({
          concept: loaded.contrast_set.concept,
          dirty: false,
          minedExamples: [],
          contrastSet: loaded.contrast_set,
          probeId: loaded.fit.probe_id,
          selectedLayer: null,
          exampleSlug: slug,
          axisSet: null,
          yourExamples: [],
        });
        setUrl({ example: slug });
        if (sessionRef.current.messages.length === 0) {
          setGhost(null);
          showConversation(loaded, 0);
        } else {
          await colorConversation(apiBaseRef.current, loaded.fit.probe_id, loaded, ghostCandidate);
        }
      } catch (err) {
        setError("load", `Could not load the prebuilt example "${slug}": ${describeError(err)}`);
      } finally {
        setBusy(null);
      }
    },
    [captureGhost, clearDeflation, colorConversation, setError, showConversation, stopAxis, stopConcept, stopStreaming, update],
  );

  const loadProbe = useCallback(
    async (base: string, probeId: string) => {
      if (!base) {
        setError("load", "No server is configured, so a live probe cannot be loaded.");
        return;
      }
      setBusy("load");
      try {
        const info = await getProbe(base, probeId);
        const current = sessionRef.current;
        const sameProbe = current.probeId === probeId;
        stopStreaming();
        setExample(null);
        setCannedIndex(null);
        setFit(info);
        clearDeflation();
        liveProbeRef.current = probeId;
        setNotes([]);
        setMissingProbe(null);
        const ghostCandidate = sameProbe ? null : captureGhost();
        const systemPrompt = current.systemPrompt.trim() === "" ? (info.contrast_set.system_prompts[0] ?? "") : current.systemPrompt;
        const messages = current.messages;
        update({
          concept: info.concept,
          dirty: false,
          contrastSet: info.contrast_set,
          probeId,
          exampleSlug: null,
          systemPrompt,
          messages,
          selectedLayer: sameProbe ? current.selectedLayer : null,
          axisSet: sameProbe ? current.axisSet : null,
          yourExamples: sameProbe ? current.yourExamples : [],
          minedExamples: sameProbe ? current.minedExamples : [],
        });
        await colorConversation(base, probeId, null, ghostCandidate, { systemPrompt, messages });
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          setMissingProbe(probeId);
        } else {
          setError("load", `Could not load probe ${probeId}: ${describeError(err)}`);
        }
      } finally {
        setBusy(null);
      }
    },
    [captureGhost, clearDeflation, colorConversation, setError, stopStreaming, update],
  );

  const restoreSession = useCallback(
    async (base: string) => {
      const current = sessionRef.current;
      if (current.exampleSlug) {
        try {
          const loaded = await fetchExample(current.exampleSlug);
          setExample(loaded);
          setFit(loaded.fit);
          await colorConversation(base, loaded.fit.probe_id, loaded, null);
        } catch (err) {
          setError("load", `Could not reload the prebuilt example: ${describeError(err)}`);
        }
        return;
      }
      if (!current.contrastSet) {
        return;
      }
      if (!base) {
        setError("load", "The saved probe needs the server, and no server is configured.");
        return;
      }
      setBusy("load");
      try {
        const probeId = await ensureLiveProbe(base);
        await colorConversation(base, probeId, null, null);
      } catch (err) {
        setError("load", `Could not restore the saved probe: ${describeError(err)}`);
      } finally {
        setBusy(null);
      }
    },
    [colorConversation, ensureLiveProbe, setError],
  );

  // Config and the example index load before the gate, so the gate knows whether the server
  // is reachable; the URL is followed only once the gate is open.
  useEffect(() => {
    let cancelled = false;
    const boot = async () => {
      const [config, index] = await Promise.all([
        fetchConfig().catch(() => ({ apiBase: "" })),
        fetchExampleIndex().catch(() => [] as ExampleSummary[]),
      ]);
      if (cancelled) {
        return;
      }
      const base = config.apiBase.replace(/\/+$/, "");
      apiBaseRef.current = base;
      setApiBase(base);
      setExamples(index);
      examplesRef.current = index;
      setConfigReady(true);
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!unlocked || !configReady) {
      return;
    }
    const route = async () => {
      const base = apiBaseRef.current;
      const params = new URLSearchParams(window.location.search);
      const exampleParam = params.get("example");
      const probeParam = params.get("probe");
      if (exampleParam) {
        await loadExample(exampleParam);
        return;
      }
      if (probeParam) {
        await loadProbe(base, probeParam);
        return;
      }
      await restoreSession(base);
      // A first visit with nothing to show lands on the first prebuilt example, so the tour
      // has every target on the page.
      const current = sessionRef.current;
      const first = examplesRef.current[0];
      if (tourArmed.current && !current.exampleSlug && !current.contrastSet && first) {
        await loadExample(first.slug);
      }
    };
    void route();
  }, [configReady, unlocked, loadExample, loadProbe, restoreSession]);

  // The first-visit tour opens once the page has a fit and nothing is loading.
  useEffect(() => {
    if (!tourArmed.current || !unlocked || fit === null || busy !== null) {
      return;
    }
    tourArmed.current = false;
    setTourOpen(true);
  }, [busy, fit, unlocked]);

  // Replaying the tour on an empty page loads the first prebuilt example first.
  const startTour = useCallback(async () => {
    const first = examplesRef.current[0];
    if (fit === null && first && busy === null) {
      await loadExample(first.slug);
    }
    setTourOpen(true);
  }, [busy, fit, loadExample]);

  const closeTour = useCallback(() => {
    markTourSeen();
    setTourOpen(false);
  }, []);

  // Any 401 drops the stored password and brings the gate back with a note.
  useEffect(() => {
    onUnauthorized(() => {
      clearStoredKey();
      stopStreaming();
      stopConcept();
      stopAxis();
      stopMine();
      setBusy(null);
      setGateNote("The server rejected the stored password; enter it again.");
      setUnlocked(false);
    });
    return () => onUnauthorized(null);
  }, [stopAxis, stopConcept, stopMine, stopStreaming]);

  const submitPassword = useCallback(async (password: string): Promise<boolean> => {
    const base = apiBaseRef.current;
    if (!base) {
      setGateNote("No server is configured, so the password cannot be checked.");
      return false;
    }
    try {
      const ok = await checkKey(base, password);
      if (ok) {
        setStoredKey(password);
        setGateNote(null);
        setUnlocked(true);
      }
      return ok;
    } catch (err) {
      setGateNote(`The password could not be checked: ${describeError(err)}`);
      return false;
    }
  }, []);

  const continueOffline = useCallback(() => {
    setGateNote(null);
    setUnlocked(true);
  }, []);

  const signOut = useCallback(() => {
    clearStoredKey();
    stopStreaming();
    stopConcept();
    stopAxis();
    stopMine();
    setBusy(null);
    setGateNote(null);
    setUnlocked(false);
  }, [stopAxis, stopConcept, stopMine, stopStreaming]);

  const buildProbe = useCallback(
    async (concept: string) => {
      const base = apiBaseRef.current;
      if (!base) {
        return;
      }
      stopConcept();
      stopAxis();
      setErrors({});
      setMissingProbe(null);
      setBusy("concept");
      setConceptStatus("generating the contrast set, about a minute");
      const controller = new AbortController();
      conceptAbortRef.current = controller;
      let contrastSet: ContrastSet | null = null;
      try {
        let generated = null as ConceptResponse | null;
        await streamConcept(
          base,
          concept,
          (event) => {
            if (event.type === "status") {
              setConceptStatus(event.text);
            } else if (event.type === "concept") {
              generated = event;
            } else {
              throw new ApiError(0, event.error);
            }
          },
          controller.signal,
        );
        if (controller.signal.aborted) {
          return;
        }
        if (!generated) {
          throw new ApiError(0, "The server ended the stream without a contrast set.");
        }
        contrastSet = generated.contrast_set;
        stopStreaming();
        const ghostCandidate = captureGhost();
        const before = sessionRef.current;
        setExample(null);
        setCannedIndex(null);
        setFit(null);
        clearDeflation();
        setMisfires(null);
        setMineNotes([]);
        setEvidence([]);
        liveProbeRef.current = null;
        setNotes(droppedNotes(generated.dropped, contrastSet as unknown as Record<string, unknown>, LIST_LABELS));
        update({
          concept,
          contrastSet,
          dirty: false,
          minedExamples: [],
          probeId: null,
          systemPrompt: before.messages.length === 0 ? (contrastSet.system_prompts[0] ?? "") : before.systemPrompt,
          selectedLayer: null,
          exampleSlug: null,
          axisSet: null,
          yourExamples: [],
        });
        setUrl(null);
        setConceptStatus("fitting the direction");
        setBusy("fit");
        const fitted = await postFit(base, concept, contrastSet);
        applyFit(fitted, false);
        setUrl({ probe: fitted.probe_id });
        await colorConversation(base, fitted.probe_id, null, ghostCandidate);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        setError(contrastSet ? "set" : "concept", describeError(err));
      } finally {
        if (conceptAbortRef.current === controller) {
          conceptAbortRef.current = null;
          setConceptStatus(null);
          setBusy(null);
        }
      }
    },
    [applyFit, captureGhost, clearDeflation, colorConversation, setError, stopAxis, stopConcept, stopStreaming, update],
  );

  // Refits from the set. When the fit on screen has an axis projected out, the same axis is
  // projected out of the new fit too (decoys, mining, edits alike), so the badge and axis line
  // stay, evidence compares deflated with deflated, and restore points at the plain refit.
  const refit = useCallback(
    async (override?: ContrastSet): Promise<FitResponse | null> => {
      const keepAxis = fitRef.current?.deflated_axis ?? null;
      const base = apiBaseRef.current;
      const current = sessionRef.current;
      const contrastSet = override ?? current.contrastSet;
      if (!base || !contrastSet) {
        return null;
      }
      setError("set", undefined);
      setBusy("fit");
      try {
        const ghostCandidate = captureGhost();
        let fitted = await postFit(base, current.concept || contrastSet.concept, contrastSet);
        if (keepAxis) {
          setAxisStatus(`projecting ${keepAxis} out of the refit`);
          const deflated = await carryDeflation(base, fitted, keepAxis, contrastSet);
          setPreviousFit(fitted);
          recordEvidence(axisEvidence(deflated, fitted, current.axisRounds));
          fitted = deflated.fit;
        }
        applyFit(fitted, false);
        update({ exampleSlug: null });
        setUrl({ probe: fitted.probe_id });
        await colorConversation(base, fitted.probe_id, exampleRef.current, ghostCandidate);
        return fitted;
      } catch (err) {
        setError("set", describeError(err));
        return null;
      } finally {
        if (keepAxis) {
          setAxisStatus(null);
        }
        setBusy(null);
      }
    },
    [applyFit, captureGhost, carryDeflation, colorConversation, recordEvidence, setError, update],
  );

  const deleteExample = useCallback(
    (list: TrainList, index: number) => {
      update((prev) => {
        if (!prev.contrastSet) {
          return prev;
        }
        const contrastSet = { ...prev.contrastSet, [list]: prev.contrastSet[list].filter((_, i) => i !== index) };
        return { ...prev, contrastSet, exampleSlug: null, dirty: true };
      });
    },
    [update],
  );

  const addExample = useCallback(
    (list: TrainList, example: Example) => {
      update((prev) => {
        if (!prev.contrastSet) {
          return prev;
        }
        const contrastSet = { ...prev.contrastSet, [list]: [...prev.contrastSet[list], example] };
        return { ...prev, contrastSet, exampleSlug: null, yourExamples: [...prev.yourExamples, example.text], dirty: true };
      });
    },
    [update],
  );

  const addDecoys = useCallback(() => {
    const current = sessionRef.current.contrastSet;
    if (!current) {
      return;
    }
    const extra = decoysToAdd(current);
    if (extra.length === 0) {
      return;
    }
    const next: ContrastSet = { ...current, train_neg: [...current.train_neg, ...extra] };
    const before = fitRef.current?.confound ?? null;
    update({ contrastSet: next, exampleSlug: null });
    void refit(next).then((fitted) => {
      if (before && fitted) {
        recordEvidence({ kind: "decoys", label: "after adding decoys", before, after: fitted.confound });
      }
    });
  }, [recordEvidence, refit, update]);

  // Top-scoring tokens in concept-absent examples at the selected layer: what a naive probe rides on.
  const showMisfires = useCallback(async () => {
    const base = apiBaseRef.current;
    if (!base || !sessionRef.current.contrastSet) {
      return;
    }
    setError("mine", undefined);
    setBusy("misfires");
    try {
      const probeId = await ensureLiveProbe(base);
      setMisfires(await postMisfires(base, { probe_id: probeId, layer: layerRef.current, top_k: 12 }));
    } catch (err) {
      setError("mine", describeError(err));
    } finally {
      setBusy(null);
    }
  }, [ensureLiveProbe, setError]);

  const saveCurrent = useCallback(
    (name: string) => {
      const current = sessionRef.current;
      const fitted = fitRef.current;
      if (!fitted || !current.contrastSet) {
        return;
      }
      const entry: SavedProbe = {
        name,
        probe_id: fitted.probe_id,
        concept: current.concept || current.contrastSet.concept,
        deflated_axis: fitted.deflated_axis ?? null,
        contrast_set: current.contrastSet,
        axis_set: fitted.deflated_axis ? current.axisSet : null,
        axis_rounds: fitted.deflated_axis ? current.axisRounds : 1,
        fit: fitted,
        system_prompt: current.systemPrompt,
        saved_at: new Date().toISOString(),
        your_examples: current.yourExamples,
        mined_examples: current.minedExamples,
      };
      const evicted = save(entry);
      setSaveNote(evicted ? `saved as "${name}"; the list keeps 20, so the oldest, "${evicted.name}", was removed` : `saved as "${name}"`);
    },
    [save],
  );

  // Refits a saved probe from its stored set, with its axis projected out again when it had one.
  const rebuildSaved = useCallback(
    async (base: string, entry: SavedProbe): Promise<FitResponse> => {
      const refit = await postFit(base, entry.concept, entry.contrast_set);
      if (!entry.deflated_axis) {
        return refit;
      }
      const axisSet = entry.axis_set ?? (await loadAxisSet(base, entry.concept, entry.contrast_set, entry.deflated_axis)).axis_set;
      return (await postDeflate(base, deflateRequest(refit.probe_id, axisSet, entry.axis_rounds ?? 1))).fit;
    },
    [loadAxisSet],
  );

  // Loads a saved probe the way a permalink does: the server's copy when it has one, else a
  // refit from the stored set (same id) with the stored axis projected out again; offline, the
  // stored fit renders the panels. The conversation stays and recolors with ghosts.
  const loadSaved = useCallback(
    async (probeId: string) => {
      const entry = savedRef.current.find((probe) => probe.probe_id === probeId);
      if (!entry) {
        return;
      }
      const base = apiBaseRef.current;
      stopConcept();
      stopAxis();
      stopMine();
      stopStreaming();
      const ghostCandidate = captureGhost();
      setBusy("load");
      setErrors({});
      setMissingProbe(null);
      setSaveNote(null);
      try {
        let fitted: FitResponse = entry.fit;
        let contrastSet = entry.contrast_set;
        let live = false;
        if (base && statusRef.current !== "offline") {
          try {
            const info = await getProbe(base, entry.probe_id);
            fitted = info;
            contrastSet = info.contrast_set;
            live = true;
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) {
              fitted = await rebuildSaved(base, entry);
              live = true;
            } else if (!(err instanceof ApiError && err.status === 0)) {
              throw err;
            }
          }
        }
        setExample(null);
        setCannedIndex(null);
        setFit(fitted);
        clearDeflation();
        setMisfires(null);
        setMineNotes([]);
        setEvidence([]);
        setNotes([]);
        liveProbeRef.current = live ? fitted.probe_id : null;
        const current = sessionRef.current;
        update({
          concept: entry.concept,
          contrastSet,
          probeId: fitted.probe_id,
          exampleSlug: null,
          axisSet: entry.axis_set,
          axisRounds: clampRounds(entry.axis_rounds ?? 1),
          selectedLayer: null,
          dirty: false,
          yourExamples: entry.your_examples,
          minedExamples: entry.mined_examples,
          systemPrompt: current.systemPrompt.trim() === "" ? entry.system_prompt : current.systemPrompt,
        });
        setUrl({ probe: fitted.probe_id });
        await colorConversation(base, fitted.probe_id, null, ghostCandidate);
      } catch (err) {
        setError("load", `Could not load the saved probe "${entry.name}": ${describeError(err)}`);
      } finally {
        setBusy(null);
      }
    },
    [captureGhost, clearDeflation, colorConversation, rebuildSaved, setError, stopAxis, stopConcept, stopMine, stopStreaming, update],
  );

  // Streams decoys written around the misfire seeds, appends them to the negatives, and refits
  // through the normal path so the ghosts and the before-and-after read like the other fixes.
  const mineDecoys = useCallback(
    async (seeds: string[]) => {
      const base = apiBaseRef.current;
      const current = sessionRef.current;
      const contrastSet = current.contrastSet;
      if (!base || !contrastSet) {
        return;
      }
      stopMine();
      setError("mine", undefined);
      setBusy("mine");
      setMineStatus(`writing decoys around ${seeds.join(", ")}`);
      const controller = new AbortController();
      mineAbortRef.current = controller;
      try {
        let mined = null as MineResponse | null;
        await streamMine(
          base,
          { concept: current.concept || contrastSet.concept, contrast_set: contrastSet, seeds },
          (event) => {
            if (event.type === "status") {
              setMineStatus(event.text);
            } else if (event.type === "mined") {
              mined = event;
            } else {
              throw new ApiError(0, event.error);
            }
          },
          controller.signal,
        );
        if (controller.signal.aborted) {
          return;
        }
        if (!mined) {
          throw new ApiError(0, "The server ended the stream without decoys.");
        }
        const present = new Set(contrastSet.train_neg.map((example) => example.text));
        const fresh = mined.decoys.filter((decoy) => !present.has(decoy.text));
        setMineNotes([
          ...droppedNotes(mined.dropped, { decoys: mined.decoys }, MINE_LABELS),
          `${fresh.length} mined decoys added to the negatives; the held-out panel decoys are untouched`,
        ]);
        const next: ContrastSet = { ...contrastSet, train_neg: [...contrastSet.train_neg, ...fresh] };
        update({ contrastSet: next, exampleSlug: null, minedExamples: [...current.minedExamples, ...fresh.map((decoy) => decoy.text)] });
        setMineStatus(null);
        const before = fitRef.current?.confound ?? null;
        const fitted = await refit(next);
        if (before && fitted) {
          recordEvidence({ kind: "mine", label: "after mining decoys", before, after: fitted.confound });
        }
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        setError("mine", describeError(err));
      } finally {
        if (mineAbortRef.current === controller) {
          mineAbortRef.current = null;
          setMineStatus(null);
          setBusy(null);
        }
      }
    },
    [recordEvidence, refit, setError, stopMine, update],
  );

  // Scores the conversation under another probe; one the pod has lost is rebuilt when the
  // caller knows its set, otherwise the loss is reported.
  const scoreUnder = useCallback(
    async (
      base: string,
      probeId: string,
      rebuild: (() => Promise<FitResponse>) | null,
      conversation: { systemPrompt: string; messages: Message[] },
    ): Promise<ScoredTurn[]> => {
      const request = (id: string) => postScore(base, { probe_id: id, system_prompt: conversation.systemPrompt, messages: conversation.messages });
      try {
        return (await request(probeId)).turns;
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 404) || !rebuild) {
          throw err;
        }
        return (await request((await rebuild()).probe_id)).turns;
      }
    },
    [],
  );

  // Every other loaded probe: the prebuilt examples, the saved probes, and the probe the
  // ghosts came from.
  const compareTargets = useMemo<CompareTarget[]>(() => {
    const currentId = fit?.probe_id ?? null;
    const targets: CompareTarget[] = examples
      .filter((summary) => summary.slug !== session.exampleSlug)
      .map((summary) => ({ key: `example:${summary.slug}`, label: summary.title, name: summary.title, kind: "example", slug: summary.slug }));
    for (const probe of saved) {
      if (probe.probe_id !== currentId) {
        targets.push({ key: `saved:${probe.probe_id}`, label: probe.name, name: probe.name, kind: "saved", probeId: probe.probe_id });
      }
    }
    const previous = ghost && ghost.kind !== "edit" ? (ghost.probeId ?? null) : null;
    if (previous && previous !== currentId && !saved.some((probe) => probe.probe_id === previous)) {
      const name = ghost?.label ?? "probe";
      targets.push({ key: "previous", label: `previous: ${name}`, name, kind: "previous", probeId: previous });
    }
    return targets;
  }, [examples, fit, ghost, saved, session.exampleSlug]);
  const compareTargetsRef = useRef(compareTargets);
  compareTargetsRef.current = compareTargets;

  const compareWith = useCallback(
    async (key: string) => {
      const target = compareTargetsRef.current.find((candidate) => candidate.key === key);
      const base = apiBaseRef.current;
      const current = sessionRef.current;
      const conversation = { systemPrompt: current.systemPrompt, messages: current.messages };
      if (!target || conversation.messages.length === 0) {
        return;
      }
      const needsServer = () => {
        if (!base || statusRef.current === "offline") {
          throw new ApiError(0, "scoring under another probe needs the live server");
        }
        return base;
      };
      setCompareBusy(true);
      setCompareError(null);
      try {
        let turns: ScoredTurn[];
        if (target.kind === "example" && target.slug) {
          const loaded = await fetchExample(target.slug);
          const index = findCanned(loaded, conversation.systemPrompt, conversation.messages);
          if (index >= 0) {
            turns = loaded.conversations[index].scored.turns;
          } else {
            const live = needsServer();
            turns = await scoreUnder(live, loaded.fit.probe_id, () => postFit(live, loaded.contrast_set.concept, loaded.contrast_set), conversation);
          }
        } else if (target.kind === "saved" && target.probeId) {
          const entry = savedRef.current.find((probe) => probe.probe_id === target.probeId);
          if (!entry) {
            return;
          }
          const live = needsServer();
          turns = await scoreUnder(live, entry.probe_id, () => rebuildSaved(live, entry), conversation);
        } else if (target.probeId) {
          turns = await scoreUnder(needsServer(), target.probeId, null, conversation);
        } else {
          return;
        }
        setComparison({ label: target.name, turns });
      } catch (err) {
        setCompareError(
          err instanceof ApiError && err.status === 404
            ? `${target.label} is gone from the server; load it again to rebuild it.`
            : `Could not score under ${target.label}: ${describeError(err)}`,
        );
      } finally {
        setCompareBusy(false);
      }
    },
    [rebuildSaved, scoreUnder],
  );

  const closeComparison = useCallback(() => {
    setComparison(null);
    setCompareError(null);
  }, []);

  // The comparison was scored on one conversation; any change to it starts over.
  useEffect(() => {
    setComparison(null);
    setCompareError(null);
  }, [session.messages, session.systemPrompt]);

  // Streams the axis pairs (reusing the saved set when the axis name matches), projects the
  // axis out of the current probe, and swaps the deflated fit in with the layer pinned so the
  // before-and-after numbers compare like with like.
  const removeAxis = useCallback(
    async (axis: string, rounds = 1) => {
      const base = apiBaseRef.current;
      const current = sessionRef.current;
      if (!base || !current.contrastSet) {
        return;
      }
      stopAxis();
      setError("axis", undefined);
      setBusy("axis");
      setAxisStatus(`writing ${axis} pairs`);
      const controller = new AbortController();
      axisAbortRef.current = controller;
      try {
        const probeId = await ensureLiveProbe(base);
        let axisSet = current.axisSet && current.axisSet.axis === axis ? current.axisSet : null;
        if (!axisSet) {
          const generated = await loadAxisSet(base, current.concept, current.contrastSet, axis, setAxisStatus, controller.signal);
          if (controller.signal.aborted) {
            return;
          }
          axisSet = generated.axis_set;
          setAxisNotes(droppedNotes(generated.dropped, axisSet as unknown as Record<string, unknown>, AXIS_LABELS));
          update({ axisSet, contrastSet: { ...current.contrastSet, confound_axis: axis } });
        }
        setAxisStatus(rounds > 1 ? `projecting ${axisSet.axis} out over ${rounds} rounds and refitting` : `projecting ${axisSet.axis} out and refitting`);
        setBusy("deflate");
        const result = await postDeflate(base, deflateRequest(probeId, axisSet, rounds));
        if (controller.signal.aborted) {
          return;
        }
        stopStreaming();
        const ghostCandidate = captureGhost();
        const before = fitRef.current;
        setPreviousFit(before);
        if (before) {
          recordEvidence(axisEvidence(result, before, rounds));
        }
        update({ selectedLayer: layerRef.current, exampleSlug: null, axisRounds: clampRounds(rounds) });
        applyFit(result.fit, true);
        setUrl({ probe: result.fit.probe_id });
        await colorConversation(base, result.fit.probe_id, exampleRef.current, ghostCandidate);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        setError("axis", describeError(err));
      } finally {
        if (axisAbortRef.current === controller) {
          axisAbortRef.current = null;
          setAxisStatus(null);
          setBusy(null);
        }
      }
    },
    [applyFit, captureGhost, colorConversation, ensureLiveProbe, loadAxisSet, recordEvidence, setError, stopAxis, stopStreaming, update],
  );

  const restoreOriginal = useCallback(async () => {
    const original = previousFit;
    if (!original) {
      return;
    }
    stopStreaming();
    setError("axis", undefined);
    const ghostCandidate = captureGhost();
    applyFit(original, true);
    // The original probe may have left the pod meanwhile; the next call re-checks it.
    liveProbeRef.current = null;
    setUrl({ probe: original.probe_id });
    await colorConversation(apiBaseRef.current, original.probe_id, exampleRef.current, ghostCandidate);
  }, [applyFit, captureGhost, colorConversation, previousFit, setError, stopStreaming]);

  const sendMessage = useCallback(
    async (text: string): Promise<boolean> => {
      const base = apiBaseRef.current;
      const current = sessionRef.current;
      if (!base || !current.contrastSet) {
        return false;
      }
      const messages: Message[] = [...current.messages, { role: "user", content: text }];
      update({ messages });
      setError("chat", undefined);
      setBusy("chat");
      setStreamingText("");
      const controller = new AbortController();
      chatAbortRef.current = controller;
      try {
        const probeId = await ensureLiveProbe(base);
        const request: ChatRequest = {
          probe_id: probeId,
          system_prompt: current.systemPrompt,
          messages,
          alpha: current.alpha,
          layer: layerRef.current,
          max_new_tokens: current.maxNewTokens,
          temperature: 0.7,
          seed: 0,
        };
        let scored = null as ChatScored | null;
        await streamChat(
          base,
          request,
          (event) => {
            if (event.type === "token") {
              if (event.text !== "") {
                setStreamingText((prev) => (prev ?? "") + event.text);
              }
            } else if (event.type === "scored") {
              scored = event;
            } else {
              throw new ApiError(0, event.error);
            }
          },
          controller.signal,
        );
        if (!scored) {
          throw new ApiError(0, "The reply ended without a score.");
        }
        setTurns(scored.turns);
        setChatStatus(chatStatusOf(scored));
        setGhost(null);
        setCannedIndex(null);
        update({ messages: messagesFromTurns(scored.turns) });
        return true;
      } catch (err) {
        if (controller.signal.aborted) {
          return false;
        }
        update({ messages: current.messages });
        setError("chat", describeError(err));
        return false;
      } finally {
        if (chatAbortRef.current === controller) {
          chatAbortRef.current = null;
        }
        setStreamingText(null);
        setBusy(null);
      }
    },
    [ensureLiveProbe, setError, update],
  );

  // Recolors the conversation as it now stands (edited text or a changed system prompt) under
  // the current probe, the same path a probe switch uses, keeping the old scores as ghosts.
  const recolorConversation = useCallback(
    async (messages: Message[]) => {
      const current = sessionRef.current;
      const ghostCandidate = captureGhost("before edit", "edit");
      stopStreaming();
      setError("chat", undefined);
      setCannedIndex(null);
      setChatStatus(null);
      update({ messages });
      await colorConversation(apiBaseRef.current, fitRef.current?.probe_id ?? null, exampleRef.current, ghostCandidate, {
        systemPrompt: current.systemPrompt,
        messages,
      });
    },
    [captureGhost, colorConversation, setError, stopStreaming, update],
  );

  const editMessage = useCallback(
    (index: number, text: string) => {
      const current = sessionRef.current;
      if (index < 0 || index >= current.messages.length) {
        return;
      }
      void recolorConversation(current.messages.map((message, i) => (i === index ? { ...message, content: text } : message)));
    },
    [recolorConversation],
  );

  const recolorNow = useCallback(() => {
    void recolorConversation(sessionRef.current.messages);
  }, [recolorConversation]);

  // Drops the last user message and the reply after it. No network call: earlier turns'
  // activations do not depend on later ones, so their scores stand.
  const deleteLastPair = useCallback(() => {
    const current = sessionRef.current;
    const lastUser = current.messages.map((message) => message.role).lastIndexOf("user");
    if (lastUser < 0) {
      return;
    }
    stopStreaming();
    const messages = current.messages.slice(0, lastUser);
    setTurns((prev) => {
      if (messages.length === 0) {
        return [];
      }
      const system = prev.filter((turn) => turn.role === "system");
      const rest = prev.filter((turn) => turn.role !== "system").slice(0, lastUser);
      return [...system, ...rest];
    });
    setGhost(null);
    setChatStatus(null);
    setCannedIndex(null);
    setError("chat", undefined);
    update({ messages });
  }, [setError, stopStreaming, update]);

  const resetChat = useCallback(() => {
    stopStreaming();
    setTurns([]);
    setChatStatus(null);
    setGhost(null);
    setCannedIndex(null);
    setError("chat", undefined);
    update({ messages: [] });
  }, [setError, stopStreaming, update]);

  const startOver = useCallback(() => {
    stopStreaming();
    stopConcept();
    stopAxis();
    stopMine();
    reset();
    setExample(null);
    setCannedIndex(null);
    setFit(null);
    clearDeflation();
    setMisfires(null);
    setMineNotes([]);
    setEvidence([]);
    setNotes([]);
    setTurns([]);
    setChatStatus(null);
    setGhost(null);
    setErrors({});
    setMissingProbe(null);
    liveProbeRef.current = null;
    setUrl(null);
  }, [clearDeflation, reset, stopAxis, stopConcept, stopMine, stopStreaming]);

  // Loads a canned conversation in place of the current one, colored under the probe now shown.
  const pickCanned = useCallback(
    async (index: number) => {
      const loaded = exampleRef.current;
      const conversation = loaded?.conversations[index];
      if (!loaded || !conversation) {
        return;
      }
      stopStreaming();
      setError("chat", undefined);
      setGhost(null);
      const systemPrompt = conversation.system_prompt;
      const messages = messagesFromTurns(conversation.scored.turns);
      update({ systemPrompt, messages });
      setCannedIndex(index);
      await colorConversation(apiBaseRef.current, fitRef.current?.probe_id ?? null, loaded, null, { systemPrompt, messages });
    },
    [colorConversation, setError, stopStreaming, update],
  );

  const conceptBusy = busy === "concept" || busy === "fit" ? busy : null;
  const axisBusy = busy === "axis" || busy === "deflate" ? busy : null;
  const mineBusy = busy === "misfires" || busy === "mine" ? busy : null;
  const chatBusy = busy === "chat" || busy === "score";
  const defaultAxis = session.contrastSet?.confound_axis ?? DEFAULT_AXIS;
  const canSave = fit !== null && !ownExample && session.contrastSet !== null;
  let decoysState: DecoysState = "ready";
  if (session.contrastSet && decoysToAdd(session.contrastSet).length === 0) {
    decoysState = "added";
  } else if (session.contrastSet && session.contrastSet.decoys.length < DECOYS_FOR_FIX * 2) {
    decoysState = "few";
  }
  const defaultName =
    fit && session.contrastSet ? defaultProbeName(session.concept || session.contrastSet.concept, session.contrastSet, fit, session.minedExamples.length) : "";
  const permalink = session.exampleSlug
    ? `${window.location.origin}${window.location.pathname}?example=${encodeURIComponent(session.exampleSlug)}`
    : fit
      ? `${window.location.origin}${window.location.pathname}?probe=${encodeURIComponent(fit.probe_id)}`
      : null;
  const systemPromptChips = session.contrastSet ? session.contrastSet.system_prompts : [];

  let statusText = "checking the server";
  if (status === "online" && health) {
    statusText = `live: ${health.model_id}, ${health.n_layers} layers`;
  } else if (status === "offline") {
    statusText = "offline";
  }

  if (!unlocked) {
    return <Gate status={status} note={gateNote} onSubmit={submitPassword} onContinueOffline={continueOffline} />;
  }

  return (
    <>
      <header className="masthead">
        <div>
          <h1>Probe Bench</h1>
          <p className="tagline">the dataset is the probe</p>
        </div>
        <div className="status">
          <span>
            <span className={`dot${online ? " dot-online" : ""}`} />
            {statusText}
          </span>
          <button type="button" className="link" onClick={() => void startTour()} title="Replay the first-visit tour">
            tour
          </button>
          {getStoredKey() ? (
            <button type="button" className="link" onClick={signOut} title="Forget the access password on this browser">
              sign out
            </button>
          ) : null}
          <button type="button" onClick={startOver} title="Clear the saved session">
            Start over
          </button>
        </div>
      </header>
      {status === "offline" ? (
        <div className="banner" role="status">
          Live fitting and chat are offline. The prebuilt examples still work.{" "}
          {health?.loading ? "The model is still loading on the server. " : ""}
          Retrying in {retryIn} s.{" "}
          <button type="button" className="link" onClick={checkNow}>
            Retry now
          </button>
        </div>
      ) : null}
      {fit ? (
        <InstrumentBar
          fit={fit}
          layer={layer}
          concept={session.concept || fit.concept}
          minedCount={session.minedExamples.length}
          onLayerChange={(next) => update({ selectedLayer: next })}
          probeActions={
            <SaveProbeControls
              canSave={canSave}
              disabled={busy !== null}
              defaultName={defaultName}
              permalink={permalink}
              note={saveNote ?? storageNote}
              onSave={saveCurrent}
            />
          }
        />
      ) : null}
      <div className="columns">
        <div className="instrument">
          <ConceptInput
            concept={session.concept}
            online={online}
            busy={conceptBusy}
            status={conceptStatus}
            examples={examples}
            selectedSlug={session.exampleSlug}
            compact={fit !== null}
            saved={saved.map((probe) => ({ probeId: probe.probe_id, name: probe.name }))}
            selectedProbeId={session.exampleSlug ? null : session.probeId}
            error={errors.concept}
            onBuild={(concept) => void buildProbe(concept)}
            onPickExample={(slug) => void loadExample(slug)}
            onPickSaved={(probeId) => void loadSaved(probeId)}
            onRemoveSaved={removeSaved}
          />
          {busy === "load" ? <p className="hint">Loading</p> : null}
          {errors.load ? <p className="error">{errors.load}</p> : null}
          {missingProbe ? (
            <div className="notice">
              Probe {missingProbe} is not on the server any more; probes live on the pod and vanish when it restarts.{" "}
              {session.contrastSet ? (
                <button type="button" disabled={!online || busy !== null} onClick={() => void refit()}>
                  Rebuild it from the saved contrast set
                </button>
              ) : (
                "Build a new one from a concept."
              )}
            </div>
          ) : null}
          {fit ? (
            <ConfoundPanel
              fit={fit}
              layer={layer}
              defaultAxis={defaultAxis}
              axisSet={session.axisSet}
              canRestore={previousFit !== null}
              online={online}
              busy={axisBusy}
              disabled={busy !== null}
              status={axisStatus}
              notes={axisNotes}
              error={errors.axis}
              onRemoveAxis={(axis, rounds) => void removeAxis(axis, rounds)}
              onRestore={() => void restoreOriginal()}
              misfires={misfires}
              mineBusy={mineBusy}
              mineStatus={mineStatus}
              mineNotes={mineNotes}
              mineError={errors.mine}
              onShowMisfires={() => void showMisfires()}
              onMine={(seeds) => void mineDecoys(seeds)}
              decoysState={decoysState}
              decoysBusy={busy === "fit" || busy === "score"}
              onAddDecoys={addDecoys}
              evidence={evidence}
            />
          ) : null}
          {fit ? <LayerStrip fit={fit} layer={layer} onLayerChange={(next) => update({ selectedLayer: next })} /> : null}
        </div>
        <div className="monitor-column">
          <ChatPanel
            online={online}
            hasProbe={session.contrastSet !== null}
            layer={layer}
            layerNote={fit ? describeLayer(fit, layer) : null}
            deflatedAxis={fit?.deflated_axis ?? null}
            systemPrompt={session.systemPrompt}
            systemPromptChips={systemPromptChips}
            onSystemPromptChange={(value) => update({ systemPrompt: value })}
            turns={turns}
            messages={session.messages}
            streamingText={streamingText}
            busy={chatBusy}
            chatStatus={chatStatus}
            alpha={session.alpha}
            onAlphaChange={(alpha) => update({ alpha })}
            maxNewTokens={session.maxNewTokens}
            onMaxNewTokensChange={(value) => update({ maxNewTokens: value })}
            canned={example ? example.conversations.map((conversation) => ({ label: conversation.label })) : []}
            cannedIndex={cannedIndex}
            onPickCanned={(index) => void pickCanned(index)}
            ghost={ghost}
            probeLabel={probeLabel}
            error={errors.chat}
            onSend={sendMessage}
            onReset={resetChat}
            onEditMessage={editMessage}
            onDeleteLastPair={deleteLastPair}
            onRecolor={recolorNow}
            comparison={comparison}
            compareCandidates={compareTargets}
            compareBusy={compareBusy}
            compareError={compareError}
            onCompare={(key) => void compareWith(key)}
            onCloseCompare={closeComparison}
          />
        </div>
      </div>
      {session.contrastSet ? (
        <ContrastSetPanel
          concept={session.concept || session.contrastSet.concept}
          contrastSet={session.contrastSet}
          notes={notes}
          online={online}
          busy={busy === "fit" || busy === "score" || axisBusy !== null}
          dirty={session.dirty}
          error={errors.set}
          yours={session.yourExamples}
          mined={session.minedExamples}
          onDelete={deleteExample}
          onRefit={() => void refit()}
          onAddExample={addExample}
        />
      ) : null}
      {tourOpen ? <Tour onClose={closeTour} /> : null}
    </>
  );
}
