import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { formatZ, zToBackground } from "../colors";
import { ALPHA_LIMIT } from "../session";
import type { Message, ScoredTurn } from "../types";
import { Sparkline } from "./Sparkline";
import { TokenHeatmap, TokenReadout, useTokenFocus, type TokenFocus, type TokenPair } from "./TokenHeatmap";

export interface ChatStatus {
  hitLengthCap: boolean;
  flags: string[];
}

export const REPLY_LENGTHS = [200, 400, 800];
const ALPHA_MIN = -ALPHA_LIMIT;
const ALPHA_MAX = ALPHA_LIMIT;
const ALPHA_STEP = 0.25;

export interface Ghost {
  label: string;
  seqZ: number[][]; // per turn, per layer, from the probe shown before the switch
  kind?: "probe" | "edit";
  probeId?: string | null; // the probe those scores came from, offered for a side-by-side comparison
}

// The same conversation scored under another probe, kept in memory only.
export interface Comparison {
  label: string;
  turns: ScoredTurn[];
}

export interface CompareCandidate {
  key: string;
  label: string;
}

interface CompareTurn {
  label: string; // the comparison probe
  ownLabel: string; // the probe on screen
  turn: ScoredTurn;
}

// Token values pair up only when both rows hold the same split; a different tokenizer (or
// the mock) can hand back another one, and then each row reads alone.
function pairFor(own: ScoredTurn, other: ScoredTurn | undefined, ownFirst: boolean): TokenPair | undefined {
  if (!other || other.tokens.length !== own.tokens.length) {
    return undefined;
  }
  return { z: other.z, ownFirst };
}

interface Props {
  ghost: Ghost | null;
  online: boolean;
  hasProbe: boolean;
  layer: number;
  layerNote: string | null;
  deflatedAxis: string | null;
  systemPrompt: string;
  systemPromptChips: string[];
  onSystemPromptChange: (value: string) => void;
  turns: ScoredTurn[];
  messages: Message[];
  streamingText: string | null;
  busy: boolean;
  chatStatus: ChatStatus | null;
  alpha: number;
  onAlphaChange: (alpha: number) => void;
  maxNewTokens: number;
  onMaxNewTokensChange: (value: number) => void;
  canned: { label: string }[];
  cannedIndex: number | null;
  onPickCanned: (index: number) => void;
  error?: string;
  onSend: (text: string) => Promise<boolean>;
  onReset: () => void;
  onEditMessage: (index: number, text: string) => void;
  onDeleteLastPair: () => void;
  onRecolor: () => void;
  probeLabel: string; // the probe on screen, as the ghosts name it
  comparison: Comparison | null;
  compareCandidates: CompareCandidate[];
  compareBusy: boolean;
  compareError: string | null;
  onCompare: (key: string) => void;
  onCloseCompare: () => void;
}

// One coloring of a turn under one probe: a small muted label and chip over the tokens. The
// current probe's sub-row comes first, the comparison's under a dashed rule.
function SubRow({
  label,
  turn,
  layer,
  dim,
  focus,
  onFocusChange,
  id,
  pair,
}: {
  label: string;
  turn: ScoredTurn;
  layer: number;
  dim?: boolean;
  focus: TokenFocus | null;
  onFocusChange: (focus: TokenFocus | null) => void;
  id: string;
  pair: TokenPair | undefined;
}) {
  return (
    <div className="sub-row">
      <div className="sub-head">
        <span className="sub-label">{label}</span>
        <ScoreChip value={turn.seq_z[layer] ?? 0} small />
      </div>
      <TokenHeatmap id={id} tokens={turn.tokens} z={turn.z} layer={layer} dim={dim} focus={focus} onFocusChange={onFocusChange} pair={pair} />
    </div>
  );
}

function ScoreChip({ value, small = false }: { value: number; small?: boolean }) {
  return (
    <span
      className={`chip score${small ? " chip-small" : ""}`}
      style={{ backgroundColor: zToBackground(value) }}
      title="mean score of this turn's tokens at the selected layer"
    >
      {formatZ(value)}
    </span>
  );
}

// A scored user or assistant turn: editable inline (edit link or double-click; Enter or Save
// commits, Escape cancels) and, for the last user message, deletable together with its reply.
function ScoredTurnView({
  turn,
  layer,
  ghostNote,
  deletable,
  busy,
  compare,
  onEdit,
  onDelete,
}: {
  turn: ScoredTurn;
  layer: number;
  ghostNote: string | null;
  deletable: boolean;
  busy: boolean;
  compare: CompareTurn | null;
  onEdit: (text: string) => void;
  onDelete: () => void;
}) {
  const [focus, setFocus] = useTokenFocus();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(turn.text);
  const pair = pairFor(turn, compare?.turn, true);
  // Unpaired rows read their own token in the head when the pointer is on the second row.
  const readoutTurn = focus?.source === "other" && compare && !pair ? compare.turn : turn;

  const startEdit = () => {
    if (busy) {
      return;
    }
    setDraft(turn.text);
    setEditing(true);
  };
  const commit = () => {
    const text = draft.trim();
    setEditing(false);
    if (text !== "" && text !== turn.text) {
      onEdit(text);
    }
  };
  const onEditKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      commit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setEditing(false);
    }
  };

  return (
    <div className={`turn turn-${turn.role}`}>
      <div className="turn-head">
        <span className="role">{turn.role}</span>
        {compare ? null : <ScoreChip value={turn.seq_z[layer] ?? 0} />}
        {ghostNote ? <span className="ghost">{ghostNote}</span> : null}
        <TokenReadout tokens={readoutTurn.tokens} z={readoutTurn.z} layer={layer} focus={focus} pair={readoutTurn === turn ? pair : undefined} />
        <span className="turn-tools">
          {editing ? null : (
            <button type="button" className="link" disabled={busy} onClick={startEdit}>
              edit
            </button>
          )}
          {deletable ? (
            <button type="button" className="link" disabled={busy} title="Remove this message and the reply after it" onClick={onDelete}>
              delete
            </button>
          ) : null}
        </span>
      </div>
      {editing ? (
        <div className="turn-edit">
          <textarea value={draft} aria-label={`Edit the ${turn.role} message`} autoFocus onChange={(event) => setDraft(event.target.value)} onKeyDown={onEditKey} />
          <div className="actions">
            <button type="button" className="primary" disabled={draft.trim() === ""} onClick={commit}>
              Save
            </button>
            <button type="button" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <span className="hint">Enter saves, Shift+Enter adds a line, Escape cancels</span>
          </div>
        </div>
      ) : (
        <div className="turn-body" title="Double-click to edit" onDoubleClick={startEdit}>
          {compare ? (
            <>
              <SubRow label={compare.ownLabel} turn={turn} layer={layer} focus={focus} onFocusChange={setFocus} id="own" pair={pair} />
              <SubRow
                label={compare.label}
                turn={compare.turn}
                layer={layer}
                focus={focus}
                onFocusChange={setFocus}
                id="other"
                pair={pairFor(compare.turn, turn, false)}
              />
            </>
          ) : (
            <TokenHeatmap id="own" tokens={turn.tokens} z={turn.z} layer={layer} focus={focus} onFocusChange={setFocus} pair={pair} />
          )}
        </div>
      )}
    </div>
  );
}

function SystemTurnView({
  turn,
  layer,
  ghostNote,
  compare,
}: {
  turn: ScoredTurn;
  layer: number;
  ghostNote: string | null;
  compare: CompareTurn | null;
}) {
  const [focus, setFocus] = useTokenFocus();
  const [open, setOpen] = useState(false);
  const pair = pairFor(turn, compare?.turn, true);
  const readoutTurn = focus?.source === "other" && compare && !pair ? compare.turn : turn;
  return (
    <details className="turn turn-system" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span className="role">system</span>
        {compare ? null : <ScoreChip value={turn.seq_z[layer] ?? 0} />}
        {ghostNote ? <span className="ghost">{ghostNote}</span> : null}
        {focus ? (
          <TokenReadout tokens={readoutTurn.tokens} z={readoutTurn.z} layer={layer} focus={focus} pair={readoutTurn === turn ? pair : undefined} />
        ) : (
          <span className="muted">
            {turn.tokens.length} tokens, click to {open ? "collapse" : "expand"}
          </span>
        )}
      </summary>
      <div className="turn-body">
        {compare ? (
          <>
            <SubRow label={compare.ownLabel} turn={turn} layer={layer} dim focus={focus} onFocusChange={setFocus} id="own" pair={pair} />
            <SubRow
              label={compare.label}
              turn={compare.turn}
              layer={layer}
              dim
              focus={focus}
              onFocusChange={setFocus}
              id="other"
              pair={pairFor(compare.turn, turn, false)}
            />
          </>
        ) : (
          <TokenHeatmap id="own" tokens={turn.tokens} z={turn.z} layer={layer} dim focus={focus} onFocusChange={setFocus} pair={pair} />
        )}
      </div>
    </details>
  );
}

const CHIP_LABELS = ["raise", "lower"];

export function ChatPanel({
  online,
  hasProbe,
  layer,
  layerNote,
  deflatedAxis,
  systemPrompt,
  systemPromptChips,
  onSystemPromptChange,
  turns,
  messages,
  streamingText,
  busy,
  chatStatus,
  alpha,
  onAlphaChange,
  maxNewTokens,
  onMaxNewTokensChange,
  canned,
  cannedIndex,
  onPickCanned,
  ghost,
  error,
  onSend,
  onReset,
  onEditMessage,
  onDeleteLastPair,
  onRecolor,
  probeLabel,
  comparison,
  compareCandidates,
  compareBusy,
  compareError,
  onCompare,
  onCloseCompare,
}: Props) {
  const [draft, setDraft] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  const systemTurn = turns.find((turn) => turn.role === "system") ?? null;
  const scoredTurns = turns.filter((turn) => turn.role !== "system");
  // Ghosts only line up while the conversation is the one they were scored on.
  const ghostNoteFor = (turn: ScoredTurn): string | null => {
    if (!ghost || ghost.seqZ.length !== turns.length) {
      return null;
    }
    const row = ghost.seqZ[turns.indexOf(turn)];
    if (!row) {
      return null;
    }
    return ghost.kind === "edit" ? `was ${formatZ(row[layer] ?? 0)} before edit` : `was ${formatZ(row[layer] ?? 0)} under ${ghost.label}`;
  };
  // The comparison lines up with the turns only while it was scored on this very conversation.
  const aligned = comparison !== null && comparison.turns.length === turns.length;
  const compareTurnFor = (turn: ScoredTurn): CompareTurn | null => {
    if (!comparison || !aligned) {
      return null;
    }
    const other = comparison.turns[turns.indexOf(turn)];
    return other && other.role === turn.role ? { label: comparison.label, ownLabel: probeLabel, turn: other } : null;
  };
  const lastUserIndex = messages.map((message) => message.role).lastIndexOf("user");
  const needsServer = !online && messages.length > 0 && turns.length === 0 && streamingText === null;
  const pending = messages.slice(scoredTurns.length);
  const promptStale = systemTurn ? systemTurn.text !== systemPrompt : turns.length > 0 && systemPrompt.trim() !== "";
  const scores = turns.map((turn) => turn.seq_z[layer] ?? 0);
  const scale = Math.max(1.5, ...scores.map((value) => Math.abs(value)));
  const empty = turns.length === 0 && pending.length === 0 && streamingText === null;
  const canSend = online && hasProbe && !busy && draft.trim() !== "";

  // The page is the only scroller: follow the newest turn while it is on screen, and leave
  // the page alone when the reader has scrolled elsewhere.
  useEffect(() => {
    const last = listRef.current?.lastElementChild;
    if (!(last instanceof HTMLElement)) {
      return;
    }
    const rect = last.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      last.scrollIntoView({ block: "nearest" });
    }
  }, [streamingText, turns, pending.length]);

  const submit = async () => {
    if (!canSend) {
      return;
    }
    const ok = await onSend(draft.trim());
    if (ok) {
      setDraft("");
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  };

  let blocker: string | null = null;
  if (!online) {
    blocker = "Sending is off while the server is offline.";
  } else if (!hasProbe) {
    blocker = "Build a probe or pick a prebuilt example first.";
  }

  return (
    <section className="panel panel-first monitor" data-tour="monitor">
      <div className="panel-head">
        <h2>Monitor</h2>
        <span className="muted">
          {layerNote ?? "no probe loaded yet"}
          {deflatedAxis ? <span className="badge badge-inline">deflated: {deflatedAxis}</span> : null}
          {turns.length > 0 || comparison ? (
            <span className="compare-control">
              {comparison ? (
                <>
                  {probeLabel} vs {comparison.label}{" "}
                  <button type="button" className="link" onClick={onCloseCompare}>
                    close
                  </button>
                </>
              ) : compareBusy ? (
                "scoring under the other probe"
              ) : (
                <label>
                  compare with{" "}
                  <select
                    value=""
                    aria-label="Compare with another probe"
                    disabled={busy || compareCandidates.length === 0}
                    onChange={(event) => {
                      if (event.target.value !== "") {
                        onCompare(event.target.value);
                      }
                    }}
                  >
                    <option value="">{compareCandidates.length === 0 ? "no other probe loaded" : "choose a probe"}</option>
                    {compareCandidates.map((candidate) => (
                      <option key={candidate.key} value={candidate.key}>
                        {candidate.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </span>
          ) : null}
        </span>
      </div>
      {compareError ? <p className="error">{compareError}</p> : null}
      <details className="collapsible how">
        <summary>How to read this</summary>
        <p className="hint">
          Chat with the model under a system prompt you can edit. Each turn is scored by the direction: orange is more of the concept, blue is
          less, in units of the training spread. Turn scores use the spread of whole training examples, the same units as the panel bars; token
          colors use the wider per-token spread, so a single token reads smaller than its turn. Hover a token for its value; click to pin it.
        </p>
      </details>

      {canned.length > 0 ? (
        <div className="chips" aria-label="Prebuilt conversations">
          <span className="chips-label">prebuilt conversations</span>
          {canned.map((conversation, index) => (
            <button
              key={index}
              type="button"
              className={`chip${index === cannedIndex ? " chip-selected" : ""}`}
              disabled={busy}
              onClick={() => onPickCanned(index)}
            >
              {conversation.label}
            </button>
          ))}
        </div>
      ) : null}

      <div className="system-prompt">
        <div className="chips">
          <span className="chips-label">system prompt</span>
          {systemPromptChips.map((prompt, index) => (
            <button
              key={index}
              type="button"
              className={`chip${prompt === systemPrompt ? " chip-selected" : ""}`}
              title={prompt}
              disabled={busy}
              onClick={() => onSystemPromptChange(prompt)}
            >
              {CHIP_LABELS[index] ?? `prompt ${index + 1}`}
            </button>
          ))}
        </div>
        <textarea
          value={systemPrompt}
          aria-label="System prompt"
          placeholder="System prompt (optional)"
          disabled={busy}
          onChange={(event) => onSystemPromptChange(event.target.value)}
        />
        {promptStale ? (
          <span className="stale">
            The system prompt changed. Send a message to recolor the conversation under it, or{" "}
            <button type="button" className="link" disabled={busy || !online} onClick={onRecolor}>
              recolor now
            </button>
            .
          </span>
        ) : null}
      </div>

      {turns.length > 0 ? (
        <div className="conversation-strip">
          <Sparkline values={scores} domain={[-scale, scale]} baseline={0} markers height={56} label="Turn scores across the conversation" />
          <div className="readout">
            turn scores at layer {layer}
            {comparison && aligned ? ` under ${probeLabel}` : ""}:{" "}
            {turns.map((turn, index) => (
              <span key={index}>
                {index > 0 ? ", " : ""}
                {turn.role} {formatZ(turn.seq_z[layer] ?? 0)}
              </span>
            ))}
          </div>
          {comparison && aligned ? (
            <div className="readout">
              under {comparison.label}:{" "}
              {comparison.turns.map((turn, index) => (
                <span key={index}>
                  {index > 0 ? ", " : ""}
                  {turn.role} {formatZ(turn.seq_z[layer] ?? 0)}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="messages" ref={listRef}>
        {needsServer ? <p className="hint">colors for this conversation need the live server</p> : null}
        {systemTurn ? <SystemTurnView turn={systemTurn} layer={layer} ghostNote={ghostNoteFor(systemTurn)} compare={compareTurnFor(systemTurn)} /> : null}
        {scoredTurns.map((turn, index) => (
          <ScoredTurnView
            key={index}
            turn={turn}
            layer={layer}
            ghostNote={ghostNoteFor(turn)}
            deletable={turn.role === "user" && index === lastUserIndex}
            busy={busy}
            compare={compareTurnFor(turn)}
            onEdit={(text) => onEditMessage(index, text)}
            onDelete={onDeleteLastPair}
          />
        ))}
        {pending.map((message, index) => (
          <div key={`pending-${index}`} className={`turn turn-${message.role}`}>
            <div className="turn-head">
              <span className="role">{message.role}</span>
              <span className="chip chip-muted">{busy ? "scored after the reply" : "not scored yet"}</span>
              {message.role === "user" && scoredTurns.length + index === lastUserIndex && !busy ? (
                <span className="turn-tools">
                  <button type="button" className="link" title="Remove this message" onClick={onDeleteLastPair}>
                    delete
                  </button>
                </span>
              ) : null}
            </div>
            <div className="turn-body plain">{message.content}</div>
          </div>
        ))}
        {streamingText !== null ? (
          <div className="turn turn-assistant">
            <div className="turn-head">
              <span className="role">assistant</span>
              <span className="chip chip-muted">{streamingText === "" ? "waiting for the first token" : "streaming"}</span>
            </div>
            <div className="turn-body plain">
              {streamingText}
              <span className="caret" />
            </div>
          </div>
        ) : null}
        {empty ? <p className="empty">No conversation yet. Pick a prebuilt example or build a probe, then send a message.</p> : null}
      </div>

      {chatStatus && (chatStatus.hitLengthCap || chatStatus.flags.length > 0) ? (
        <div className="chips">
          {chatStatus.hitLengthCap ? <span className="chip chip-warn">cut at the length limit</span> : null}
          {chatStatus.flags.map((flag) => (
            <span key={flag} className="chip chip-warn">
              {flag.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      ) : null}

      <div className="composer">
        <textarea
          value={draft}
          aria-label="Message"
          placeholder={blocker ?? "Say something to the model (Enter to send, Shift+Enter for a new line)"}
          disabled={busy || blocker !== null}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="button" className="primary" disabled={!canSend} onClick={() => void submit()}>
          {busy ? "Working" : "Send"}
        </button>
        <button type="button" disabled={empty && !busy} onClick={onReset}>
          Reset
        </button>
      </div>
      {error ? <p className="error">{error}</p> : null}
      {blocker && !empty ? <p className="hint">{blocker}</p> : null}

      <div className="controls">
        <div className="control">
          <span className="control-label">reply length</span>
          {REPLY_LENGTHS.map((length) => (
            <button
              key={length}
              type="button"
              className={`chip${length === maxNewTokens ? " chip-selected" : ""}`}
              disabled={busy}
              onClick={() => onMaxNewTokensChange(length)}
            >
              {length}
            </button>
          ))}
        </div>
        <div className="control steer">
          <label htmlFor="alpha-slider">push the direction</label>
          <input
            id="alpha-slider"
            type="range"
            min={ALPHA_MIN}
            max={ALPHA_MAX}
            step={ALPHA_STEP}
            value={alpha}
            disabled={busy}
            onChange={(event) => onAlphaChange(Number(event.target.value))}
          />
          <output htmlFor="alpha-slider">{formatZ(alpha, 2)}</output>
          <span className={`slider-link${alpha === 0 ? " is-hidden" : ""}`}>
            <button type="button" className="link" onClick={() => onAlphaChange(0)}>
              off
            </button>
          </span>
        </div>
      </div>
      <p className="hint">
        Steering adds {formatZ(alpha, 2)} times the direction{hasProbe ? ` at layer ${layer}` : ""} while the reply is generated, and only then:
        the colors read the finished text with the push removed. Expect detection to be far stronger than steering, and expect the text to fall
        apart near the ends of the range. The seed is fixed, so a change between sends is the push.
      </p>
    </section>
  );
}
