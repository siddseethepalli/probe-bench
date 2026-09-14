import { useEffect, useState, type FormEvent } from "react";
import { formatPlain, formatZ, zToBackground, zToSolid } from "../colors";
import type { AxisPair, AxisSet, Confound, DeflateRound, FitResponse, Group, MisfiresResponse } from "../types";
import { clampRounds, ROUNDS_MAX, ROUNDS_MIN } from "../session";
import { bestConfoundLayer } from "./LayerStrip";

export type AxisBusy = "axis" | "deflate" | null;
export type MineBusy = "misfires" | "mine" | null;
export type DecoysState = "ready" | "added" | "few";

// Before-and-after numbers for one applied fix, per layer, read at the viewing layer.
export interface FixEvidence {
  kind: "decoys" | "axis" | "mine";
  label: string;
  before: Confound;
  after: Confound;
  axis?: { name: string; before: number[]; after: number[]; heldoutBefore?: number[]; history?: DeflateRound[]; requested?: number; applied?: number };
}

const ROUND_CHOICES = Array.from({ length: ROUNDS_MAX - ROUNDS_MIN + 1 }, (_, i) => ROUNDS_MIN + i);

interface Props {
  fit: FitResponse;
  layer: number;
  online: boolean;
  disabled: boolean;
  defaultAxis: string;
  axisSet: AxisSet | null;
  canRestore: boolean;
  busy: AxisBusy;
  status: string | null;
  notes: string[];
  error?: string;
  onRemoveAxis: (axis: string, rounds: number) => void;
  onRestore: () => void;
  misfires: MisfiresResponse | null;
  mineBusy: MineBusy;
  mineStatus: string | null;
  mineNotes: string[];
  mineError?: string;
  onShowMisfires: () => void;
  onMine: (seeds: string[]) => void;
  decoysState: DecoysState;
  decoysBusy: boolean;
  onAddDecoys: () => void;
  evidence: FixEvidence[];
}

const AXIS_MAX_CHARS = 40;

const GROUPS: { key: Group; label: string; gloss: string; inSample?: boolean }[] = [
  { key: "explicit", label: "explicit positives", gloss: "concept present, keywords present" },
  { key: "implicit", label: "implicit positives", gloss: "concept present, no keywords" },
  { key: "decoys", label: "lexical decoys", gloss: "keywords present, concept absent" },
  { key: "neutral", label: "neutral", gloss: "unrelated everyday text, training data, shown for scale", inSample: true },
];

function verdictPhrase(verdict: string): string {
  if (verdict === "tracks the concept") {
    return "it tracks the concept, not the words";
  }
  if (verdict === "mixed") {
    return "the evidence is mixed";
  }
  return "it tracks the words";
}

function evidenceLine(item: FixEvidence, layer: number): string {
  const two = (values: number[]) => formatPlain(values[layer] ?? 0, 2);
  const one = (values: number[]) => formatPlain(values[layer] ?? 0, 1);
  let line =
    `${item.label}: confound index ${two(item.before.index)} before, ${two(item.after.index)} after ` +
    `(implicit ${one(item.before.means.implicit)} to ${one(item.after.means.implicit)}, decoys ${one(item.before.means.decoys)} to ${one(item.after.means.decoys)}) at layer ${layer}`;
  if (item.axis) {
    line += `; ${item.axis.name} separability ${two(item.axis.before)} before, ${two(item.axis.after)} after`;
  }
  return line;
}

// One row per deflation round at the viewing layer, with the untouched fit as row 0; the server's
// per-round notes (which estimator, why the loop stopped) sit under the table.
function RoundTable({ item, layer }: { item: FixEvidence; layer: number }) {
  const axis = item.axis;
  if (!axis?.history) {
    return null;
  }
  const two = (values: number[] | undefined) => (values ? formatPlain(values[layer] ?? 0, 2) : "");
  const notes = axis.history.filter((round) => round.note).map((round) => `round ${round.round}: ${round.note}`);
  const short =
    axis.applied !== undefined && axis.requested !== undefined && axis.applied < axis.requested
      ? `${axis.applied} of the ${axis.requested} rounds asked for were applied`
      : null;
  return (
    <>
      <table className="round-table">
        <thead>
          <tr>
            <th>round, at layer {layer}</th>
            <th>{axis.name} separability</th>
            <th>concept held-out AUROC</th>
            <th>confound index</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>0, before</td>
            <td>{two(axis.before)}</td>
            <td>{two(axis.heldoutBefore)}</td>
            <td>{two(item.before.index)}</td>
          </tr>
          {axis.history.map((round) => (
            <tr key={round.round}>
              <td>{round.round}</td>
              <td>{two(round.axis_cv_auroc)}</td>
              <td>{two(round.heldout_auroc)}</td>
              <td>{two(round.confound_index)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {short ? <p className="round-note">{short}</p> : null}
      {notes.map((note) => (
        <p key={note} className="round-note">
          {note}
        </p>
      ))}
    </>
  );
}

function PairList({ title, axis, pairs }: { title: string; axis: string; pairs: AxisPair[] }) {
  return (
    <div className="pairs">
      <p className="pairs-title">{title}</p>
      <div className="pair-row pair-head">
        <span>high {axis}</span>
        <span>low {axis}</span>
      </div>
      {pairs.length === 0 ? <p className="muted">no pairs</p> : null}
      {pairs.map((pair, i) => (
        <div key={i} className="pair">
          {pair.context ? <div className="example-context">{pair.context}</div> : null}
          <div className="pair-row">
            <span>{pair.high}</span>
            <span>{pair.low}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function ConfoundPanel({
  fit,
  layer,
  online,
  disabled,
  defaultAxis,
  axisSet,
  canRestore,
  busy,
  status,
  notes,
  error,
  onRemoveAxis,
  onRestore,
  misfires,
  mineBusy,
  mineStatus,
  mineNotes,
  mineError,
  onShowMisfires,
  onMine,
  decoysState,
  decoysBusy,
  onAddDecoys,
  evidence,
}: Props) {
  const [open, setOpen] = useState<Group | null>(null);
  const [axisDraft, setAxisDraft] = useState(defaultAxis);
  const [rounds, setRounds] = useState(ROUNDS_MIN);
  const [hoveredMisfire, setHoveredMisfire] = useState<number | null>(null);

  useEffect(() => {
    setAxisDraft(defaultAxis);
  }, [defaultAxis]);

  const means = fit.confound.means;
  const at = (group: Group) => means[group][layer] ?? 0;
  const scale = Math.max(1.5, ...GROUPS.map((group) => Math.abs(at(group.key))));
  const verdict = fit.verdict_by_layer[layer] ?? "mixed";
  const index = fit.confound.index[layer] ?? 0;
  const isDefault = layer === bestConfoundLayer(fit);
  const deflatedAxis = fit.deflated_axis ?? null;
  const ready = online && !disabled;

  const axis = axisDraft.trim();
  let axisLabel = `Fit and remove ${axis === "" ? "the axis" : axis}`;
  if (busy === "axis") {
    axisLabel = `Writing ${axis} pairs`;
  } else if (busy === "deflate") {
    axisLabel = rounds > 1 ? `Removing ${axis} over ${rounds} rounds and refitting` : `Removing ${axis} and refitting`;
  }
  const submitAxis = (event: FormEvent) => {
    event.preventDefault();
    if (ready && axis !== "") {
      onRemoveAxis(axis, rounds);
    }
  };

  let decoysLabel = "Add decoys as negatives";
  if (decoysBusy) {
    decoysLabel = "Fitting";
  } else if (decoysState === "added") {
    decoysLabel = "Decoys already added";
  } else if (decoysState === "few") {
    decoysLabel = "Too few decoys to add";
  }

  return (
    <section className="panel panel-first verdict-panel" data-tour="verdict">
      <div className="panel-head">
        <h2>Does it track the concept or the words?</h2>
        <span className="muted">
          judging layer {layer}
          {layer === fit.picked_layer ? ", the accuracy pick" : `, not the accuracy pick ${fit.picked_layer}`}
          {deflatedAxis ? <span className="badge badge-inline">deflated: {deflatedAxis}</span> : null}
        </span>
      </div>
      {verdict === "not tested" ? (
        <p className="verdict">not tested: the stress set is empty.</p>
      ) : (
        <p className="verdict">
          At layer {layer} the direction scores implicit positives {formatPlain(at("implicit"), 1)} and decoys {formatPlain(at("decoys"), 1)}:{" "}
          {verdictPhrase(verdict)}.
        </p>
      )}
      <div className="bars">
        {GROUPS.map((group) => {
          const value = at(group.key);
          const width = (Math.abs(value) / scale) * 50;
          const left = value < 0 ? 50 - width : 50;
          const examples = fit.stress_examples.filter((example) => example.group === group.key);
          const isOpen = open === group.key;
          return (
            <div key={group.key} className={`bar-row${isOpen ? " bar-row-open" : ""}${group.inSample ? " bar-row-insample" : ""}`}>
              <button type="button" className="bar-button" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : group.key)}>
                <span className="bar-label">
                  {group.label}
                  <span className="muted"> {group.gloss}</span>
                </span>
                <span className="bar-track">
                  <span className="bar-zero" />
                  <span className="bar-fill" style={{ left: `${left}%`, width: `${width}%`, backgroundColor: zToSolid(value) }} />
                </span>
                <span className="bar-value">{formatZ(value)}</span>
              </button>
              {isOpen ? (
                <ul className="examples stress">
                  {examples.length === 0 ? <li className="muted">no examples in this group</li> : null}
                  {examples.map((example, i) => {
                    const z = example.seq_z[layer] ?? 0;
                    return (
                      <li key={i} className="example">
                        {example.context ? <div className="example-context">{example.context}</div> : null}
                        <div className="example-row">
                          <span className="chip score" style={{ backgroundColor: zToBackground(z) }}>
                            {formatZ(z)}
                          </span>
                          <span className="example-text">{example.text}</span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="hint">
        {verdict === "not tested"
          ? "The verdict needs implicit positives and held-out decoys; add them back or rebuild the set."
          : `Confound index ${formatPlain(index, 2)} (implicit minus decoys): above 1.0 reads as the concept, below 0.3 as the words.${
              isDefault ? " This default layer was chosen on these same stress sets, so it reads slightly optimistic." : ""
            }`}
      </p>

      <div className="actions fixes" data-tour="fixes">
        <button type="button" disabled={!ready || decoysState !== "ready"} onClick={onAddDecoys} title="Appends the first six decoys to the negatives and refits">
          {decoysLabel}
        </button>
        {deflatedAxis ? (
          canRestore ? (
            <button type="button" className="link" disabled={disabled} onClick={onRestore}>
              restore original direction
            </button>
          ) : (
            <span className="hint">{deflatedAxis} projected out; every refit projects it out again</span>
          )
        ) : (
          <form className="axis-inline" onSubmit={submitAxis}>
            <input
              type="text"
              value={axisDraft}
              maxLength={AXIS_MAX_CHARS}
              aria-label="Confound axis"
              disabled={busy !== null}
              onChange={(event) => setAxisDraft(event.target.value)}
            />
            <label className="rounds-select" title="Repeats the removal: each round re-estimates the tone on what the previous round left and projects that out too">
              rounds
              <select value={rounds} aria-label="Deflation rounds" disabled={busy !== null} onChange={(event) => setRounds(clampRounds(Number(event.target.value)))}>
                {ROUND_CHOICES.map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
              </select>
            </label>
            <button type="submit" disabled={!ready || axis === ""}>
              {axisLabel}
            </button>
          </form>
        )}
        <button type="button" disabled={!ready} onClick={onShowMisfires}>
          {mineBusy === "misfires" ? "Scoring the examples" : `Show misfires at layer ${layer}`}
        </button>
        <button
          type="button"
          disabled={!ready || !misfires || misfires.seeds.length === 0}
          onClick={() => {
            if (misfires) {
              onMine(misfires.seeds);
            }
          }}
        >
          {mineBusy === "mine" ? "Writing decoys" : "Write decoys around these and refit"}
        </button>
      </div>
      {evidence.map((item) => (
        <div key={item.kind} className="readout">
          {evidenceLine(item, layer)}
          {item.axis?.history && item.axis.history.length > 0 ? <RoundTable item={item} layer={layer} /> : null}
        </div>
      ))}
      {status ? (
        <p className="progress" role="status">
          <span className="pulse" />
          {status}
        </p>
      ) : null}
      {mineStatus ? (
        <p className="progress" role="status">
          <span className="pulse" />
          {mineStatus}
        </p>
      ) : null}
      {[...notes, ...mineNotes].map((note) => (
        <p key={note} className="hint">
          {note}
        </p>
      ))}
      {error ? <p className="error">{error}</p> : null}
      {mineError ? <p className="error">{mineError}</p> : null}
      {!online ? <p className="hint">The fixes need the live server.</p> : null}
      {misfires ? (
        <div className="misfires">
          <div className="readout">
            misfires at layer {misfires.layer}; seeds: {misfires.seeds.join(", ")}
          </div>
          <div className="misfire-chips">
            {misfires.misfires.map((misfire, i) => (
              <span
                key={i}
                className="chip misfire"
                style={{ backgroundColor: zToBackground(misfire.z) }}
                onMouseEnter={() => setHoveredMisfire(i)}
                onMouseLeave={() => setHoveredMisfire((current) => (current === i ? null : current))}
              >
                {misfire.token.trim() === "" ? "␣" : misfire.token.trim()} <span className="muted">{formatZ(misfire.z)}</span>
                {hoveredMisfire === i ? (
                  <span className="tok-tip tok-tip-below" role="tooltip">
                    {misfire.group}: {misfire.text}
                  </span>
                ) : null}
              </span>
            ))}
          </div>
        </div>
      ) : null}
      {axisSet ? (
        <details className="collapsible">
          <summary>
            {axisSet.axis} pairs: {axisSet.absent_pairs.length} with the concept absent, {axisSet.present_pairs.length} with it present
          </summary>
          <PairList title="concept absent" axis={axisSet.axis} pairs={axisSet.absent_pairs} />
          <PairList title="concept present" axis={axisSet.axis} pairs={axisSet.present_pairs} />
        </details>
      ) : null}
      <details className="collapsible how">
        <summary>How to read this</summary>
        <p className="hint">
          Each bar is the mean sequence score of a stress group at this layer, in units of the training spread; click a row for its examples.
          Explicit positives carry the concept and its keywords, implicit positives carry the concept without any keyword, lexical decoys carry
          the keywords without the concept, and the neutral row is training data shown for scale. A direction that reads the concept scores
          implicit positives high and decoys near zero.
        </p>
        <p className="hint">
          Add decoys as negatives appends the first six decoys to the training negatives and refits. Fit and remove an axis fits a direction for
          that tone from pairs that hold the concept fixed and flip only the tone, projects it out of the activations, and refits (each extra round repeats the removal on what the previous round left, and the table under the evidence line shows what every round cost the concept); a verdict
          that survives was not riding on that axis. Show misfires lists the highest-scoring tokens in examples where the concept is absent, the
          vehicle a naive probe rides; writing decoys around them and refitting shows whether the verdict moves. A fit with a tone projected out keeps that through every refit (decoys, mining, edits), and the restore link brings back the plain refit. Each fix leaves its
          before-and-after line above.
        </p>
      </details>
    </section>
  );
}
