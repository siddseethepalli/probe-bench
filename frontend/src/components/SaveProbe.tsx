import { useEffect, useState, type FormEvent, type KeyboardEvent } from "react";

interface Props {
  canSave: boolean;
  disabled: boolean;
  defaultName: string;
  permalink: string | null;
  note: string | null;
  onSave: (name: string) => void;
}

export function SaveProbeControls({ canSave, disabled, defaultName, permalink, note, onSave }: Props) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(defaultName);
  const [copied, setCopied] = useState(false);
  const [fallbackUrl, setFallbackUrl] = useState<string | null>(null);

  useEffect(() => {
    setName(defaultName);
  }, [defaultName]);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const id = window.setTimeout(() => setCopied(false), 2000);
    return () => window.clearTimeout(id);
  }, [copied]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") {
      return;
    }
    onSave(trimmed);
    setOpen(false);
  };

  // The permalink goes to the clipboard; when the browser refuses, it is shown inline instead.
  const copyLink = async () => {
    if (!permalink) {
      return;
    }
    try {
      await navigator.clipboard.writeText(permalink);
      setCopied(true);
      setFallbackUrl(null);
    } catch {
      setFallbackUrl(permalink);
    }
  };

  if (!canSave && !permalink) {
    return null;
  }
  return (
    <div className="probe-actions">
      {canSave ? (
        <button type="button" disabled={disabled} onClick={() => setOpen((current) => !current)}>
          {open ? "Cancel" : "Save probe"}
        </button>
      ) : null}
      {permalink ? (
        <button type="button" className="link" onClick={() => void copyLink()}>
          {copied ? "link copied" : "copy link"}
        </button>
      ) : null}
      {fallbackUrl ? <input type="text" readOnly value={fallbackUrl} aria-label="Permalink" onFocus={(event) => event.target.select()} /> : null}
      {open ? (
        <form className="save-form" onSubmit={submit}>
          <input
            type="text"
            value={name}
            maxLength={80}
            aria-label="Name for the saved probe"
            autoFocus
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
              if (event.key === "Escape") {
                setOpen(false);
              }
            }}
          />
          <button type="submit" className="primary" disabled={name.trim() === ""}>
            Save
          </button>
        </form>
      ) : null}
      {note ? <span className="hint">{note}</span> : null}
    </div>
  );
}
