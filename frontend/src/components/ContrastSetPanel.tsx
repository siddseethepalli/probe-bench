import { useState, type FormEvent } from "react";
import type { ContrastSet, Example } from "../types";

export type TrainList = "train_pos" | "train_neg";

export const DECOYS_FOR_FIX = 6;

interface Props {
  concept: string;
  contrastSet: ContrastSet;
  notes: string[];
  online: boolean;
  busy: boolean;
  dirty: boolean;
  error?: string;
  yours: string[];
  mined: string[];
  onDelete: (list: TrainList, index: number) => void;
  onRefit: () => void;
  onAddExample: (list: TrainList, example: Example) => void;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Mirrors the server's keyword rule: case-insensitive, the keyword is a stem that must
// start at a word boundary ("sad" matches "sadness", "wise" does not match "otherwise").
export function hasKeyword(text: string, keywords: string[]): boolean {
  return keywords.some((keyword) => {
    const stem = keyword.trim();
    if (stem === "") {
      return false;
    }
    return new RegExp(`(?:^|[^\\p{L}\\p{N}_])${escapeRegExp(stem)}`, "iu").test(text);
  });
}

export function decoysToAdd(contrastSet: ContrastSet): Example[] {
  const present = new Set(contrastSet.train_neg.map((example) => example.text));
  return contrastSet.decoys.slice(0, DECOYS_FOR_FIX).filter((example) => !present.has(example.text));
}

// Compact rows: tag, dimmed context, text, and the remove control when the list is editable.
function ExampleRows({
  examples,
  tagFor,
  onDelete,
}: {
  examples: Example[];
  tagFor: (example: Example) => string;
  onDelete?: (index: number) => void;
}) {
  if (examples.length === 0) {
    return <p className="muted">none</p>;
  }
  return (
    <ul className="rows">
      {examples.map((example, index) => {
        const tag = tagFor(example);
        return (
          <li key={`${index}-${example.text}`} className="row">
            <span className={`tag tag-${tag}`}>{tag}</span>
            <span className="row-text">
              {example.context ? <span className="row-context">{example.context}</span> : null}
              {example.text}
            </span>
            {onDelete ? (
              <button type="button" className="delete" aria-label="Remove this example" title="Remove this example" onClick={() => onDelete(index)}>
                x
              </button>
            ) : (
              <span />
            )}
          </li>
        );
      })}
    </ul>
  );
}

// The context is offered for every set: with one, the example is read as an assistant reply
// to it; without one, as standalone text.
function AddExampleForm({ list, onAdd }: { list: TrainList; onAdd: (list: TrainList, example: Example) => void }) {
  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const label = list === "train_pos" ? "positive" : "negative";

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (trimmed === "") {
      return;
    }
    onAdd(list, { text: trimmed, context: context.trim() !== "" ? context.trim() : null });
    setText("");
    setContext("");
  };

  return (
    <form className="add-example" onSubmit={submit} aria-label={`Add a ${label} example`}>
      <input
        type="text"
        value={context}
        placeholder="the message it answers (optional)"
        aria-label={`Context for the new ${label}`}
        onChange={(event) => setContext(event.target.value)}
      />
      <div className="add-example-row">
        <input type="text" value={text} placeholder={`add a ${label} example`} aria-label={`New ${label} example`} onChange={(event) => setText(event.target.value)} />
        <button type="submit" disabled={text.trim() === ""}>
          Add
        </button>
      </div>
    </form>
  );
}

export function ContrastSetPanel({ concept, contrastSet, notes, online, busy, dirty, error, yours, mined, onDelete, onRefit, onAddExample }: Props) {
  const keywords = contrastSet.keywords;
  const yourTexts = new Set(yours);
  const minedTexts = new Set(mined);
  const positiveTag = (example: Example) => (yourTexts.has(example.text) ? "yours" : hasKeyword(example.text, keywords) ? "explicit" : "implicit");
  const decoyTexts = new Set(contrastSet.decoys.map((example) => example.text));
  const negativeTag = (example: Example) => {
    if (yourTexts.has(example.text)) {
      return "yours";
    }
    if (minedTexts.has(example.text)) {
      return "mined";
    }
    return decoyTexts.has(example.text) ? "decoy" : "matched";
  };
  const explicitCount = contrastSet.train_pos.filter((example) => positiveTag(example) === "explicit").length;
  const decoyCount = contrastSet.train_neg.filter((example) => negativeTag(example) === "decoy").length;
  const minedCount = contrastSet.train_neg.filter((example) => negativeTag(example) === "mined").length;
  const matchedCount = contrastSet.train_neg.length - decoyCount - minedCount;
  const negativesSummary =
    decoyCount > 0 || minedCount > 0
      ? `${matchedCount} matched${decoyCount > 0 ? `, ${decoyCount} decoys` : ""}${minedCount > 0 ? `, ${minedCount} mined` : ""}`
      : "matched in topic and length";
  const background = contrastSet.background ?? [];
  const half = Math.ceil(background.length / 2);
  const canFit = online && !busy;
  const counts =
    `${contrastSet.train_pos.length} positives, ${contrastSet.train_neg.length} negatives, ` +
    `${contrastSet.heldout_pos.length} + ${contrastSet.heldout_neg.length} held out, ${contrastSet.implicit_pos.length} implicit, ` +
    `${contrastSet.decoys.length} decoys, ${contrastSet.neutral.length} neutral, ${background.length} background`;

  return (
    <section className="panel data-section" data-tour="data">
      <div className="panel-head">
        <h2>{concept}</h2>
        <span className="muted">{counts}</span>
      </div>
      <p className="hint">
        {contrastSet.is_response_property ? "A property of replies, so every example carries the message it answers. " : "A property of text. "}
        Keywords a naive detector would key on:{" "}
        {keywords.map((keyword) => (
          <code key={keyword} className="kw">
            {keyword}
          </code>
        ))}
      </p>
      {notes.length > 0 ? <p className="hint">{notes.join(". ")}.</p> : null}
      <div className="actions">
        <button type="button" className={dirty ? "primary" : ""} disabled={!canFit} onClick={onRefit}>
          {busy ? "Fitting" : dirty ? "Refit (set edited)" : "Refit"}
        </button>
        {!online ? <span className="hint">Refitting needs the live server.</span> : null}
        {dirty && online ? <span className="hint">The set changed; refit to update the numbers.</span> : null}
      </div>
      {error ? <p className="error">{error}</p> : null}
      <p className="hint">
        With a message, the example is read as an assistant reply to it; without one, as standalone text. Mixing the two framings inside one set
        adds a role difference the direction can pick up, so keep a set mostly one way.
      </p>
      <div className="data-columns">
        <div>
          <h3>
            {contrastSet.train_pos.length} positives{" "}
            <span className="muted">
              ({explicitCount} explicit, {contrastSet.train_pos.length - explicitCount} implicit)
            </span>
          </h3>
          <AddExampleForm list="train_pos" onAdd={onAddExample} />
          <ExampleRows examples={contrastSet.train_pos} tagFor={positiveTag} onDelete={(index) => onDelete("train_pos", index)} />
        </div>
        <div>
          <h3>
            {contrastSet.train_neg.length} negatives <span className="muted">({negativesSummary})</span>
          </h3>
          <AddExampleForm list="train_neg" onAdd={onAddExample} />
          <ExampleRows examples={contrastSet.train_neg} tagFor={negativeTag} onDelete={(index) => onDelete("train_neg", index)} />
        </div>
      </div>
      <details className="collapsible">
        <summary>
          Held-out: {contrastSet.heldout_pos.length} positives, {contrastSet.heldout_neg.length} negatives
        </summary>
        <div className="data-columns">
          <ExampleRows examples={contrastSet.heldout_pos} tagFor={() => "positive"} />
          <ExampleRows examples={contrastSet.heldout_neg} tagFor={() => "negative"} />
        </div>
      </details>
      <details className="collapsible">
        <summary>
          Stress sets: {contrastSet.implicit_pos.length} implicit positives, {contrastSet.decoys.length} decoys, {contrastSet.neutral.length} neutral
        </summary>
        <p className="hint">
          Implicit positives carry the concept without any keyword. Decoys carry the keywords without the concept. Neutral text has neither. The
          first six decoys feed the fix button; the rest stay held out for the verdict panel.
        </p>
        <div className="data-columns">
          <div>
            <ExampleRows examples={contrastSet.implicit_pos} tagFor={() => "implicit"} />
            <ExampleRows examples={contrastSet.neutral} tagFor={() => "neutral"} />
          </div>
          <ExampleRows examples={contrastSet.decoys} tagFor={() => "decoy"} />
        </div>
      </details>
      <details className="collapsible">
        <summary>Background pool: {background.length} unrelated examples</summary>
        <p className="hint">
          Same format as the training set, unrelated to the concept. The shipped direction pools these with the matched negatives; the thin lines
          in the layer charts show directions fit on either pile alone.
        </p>
        <div className="data-columns">
          <ExampleRows examples={background.slice(0, half)} tagFor={() => "background"} />
          <ExampleRows examples={background.slice(half)} tagFor={() => "background"} />
        </div>
      </details>
    </section>
  );
}
