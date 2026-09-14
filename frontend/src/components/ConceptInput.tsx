import { useEffect, useState, type FormEvent } from "react";
import type { ExampleSummary } from "../api";

const CONCEPT_MAX_CHARS = 60;

interface Props {
  concept: string;
  online: boolean;
  busy: "concept" | "fit" | null;
  status: string | null;
  examples: ExampleSummary[];
  selectedSlug: string | null;
  compact: boolean;
  saved: { probeId: string; name: string }[];
  selectedProbeId: string | null;
  onPickSaved: (probeId: string) => void;
  onRemoveSaved: (probeId: string) => void;
  error?: string;
  onBuild: (concept: string) => void;
  onPickExample: (slug: string) => void;
}

export function ConceptInput({
  concept,
  online,
  busy,
  status,
  examples,
  selectedSlug,
  compact,
  saved,
  selectedProbeId,
  error,
  onBuild,
  onPickExample,
  onPickSaved,
  onRemoveSaved,
}: Props) {
  const [draft, setDraft] = useState(concept);

  useEffect(() => {
    setDraft(concept);
  }, [concept]);

  const text = draft.trim();
  const canBuild = online && busy === null && text !== "";
  const buttonLabel = busy === "concept" ? "Writing the contrast set (about a minute)" : busy === "fit" ? "Fitting the direction" : "Build probe";

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (canBuild) {
      onBuild(text);
    }
  };

  return (
    <section className="panel panel-first" data-tour="concept">
      {compact ? null : (
        <p className="lede">
          Type a concept. The model writes short examples with and without it, a direction is fit between the two piles, and the panel below
          says whether that direction tracks the concept or just its keywords. Then chat and watch every token light up.
        </p>
      )}
      <form className="concept-form" onSubmit={submit}>
        <input
          type="text"
          value={draft}
          maxLength={CONCEPT_MAX_CHARS}
          placeholder="a concept, e.g. sarcasm, legal language, sadness"
          aria-label="Concept"
          onChange={(event) => setDraft(event.target.value)}
          disabled={busy !== null}
        />
        <button type="submit" className="primary" disabled={!canBuild}>
          {buttonLabel}
        </button>
      </form>
      {status ? (
        <p className="progress" role="status">
          <span className="pulse" />
          {status}
        </p>
      ) : null}
      {error ? <p className="error">{error}</p> : null}
      {!online ? <p className="hint">Live fitting is offline right now. The prebuilt examples below work without the server.</p> : null}
      {examples.length > 0 ? (
        <div className="chips" aria-label="Prebuilt examples">
          <span className="chips-label">prebuilt</span>
          {examples.map((example) => (
            <button
              key={example.slug}
              type="button"
              className={`chip${example.slug === selectedSlug ? " chip-selected" : ""}`}
              title={example.blurb}
              onClick={() => onPickExample(example.slug)}
            >
              {example.title}
            </button>
          ))}
        </div>
      ) : null}
      {saved.length > 0 ? (
        <div className="chips" aria-label="Saved probes">
          <span className="chips-label">yours</span>
          {saved.map((probe) => (
            <span key={probe.probeId} className={`chip chip-saved${probe.probeId === selectedProbeId ? " chip-selected" : ""}`}>
              <button type="button" className="chip-name" title={`Load ${probe.name}`} onClick={() => onPickSaved(probe.probeId)}>
                {probe.name}
              </button>
              <button
                type="button"
                className="chip-remove"
                aria-label={`Remove ${probe.name} from saved probes`}
                title="Remove from saved probes"
                onClick={() => onRemoveSaved(probe.probeId)}
              >
                x
              </button>
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
