import { useState, type FormEvent } from "react";
import type { Status } from "../health";

interface Props {
  status: Status;
  note: string | null;
  onSubmit: (password: string) => Promise<boolean>;
  onContinueOffline: () => void;
}

// Shown on load until a access password is stored; the URL the visitor asked for is kept
// and followed once the password is accepted.
export function Gate({ status, note, onSubmit, onContinueOffline }: Props) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [rejected, setRejected] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy || password === "") {
      return;
    }
    setBusy(true);
    setRejected(false);
    try {
      const ok = await onSubmit(password);
      if (!ok) {
        setRejected(true);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="masthead">
        <div>
          <h1>Probe Bench</h1>
          <p className="tagline">the dataset is the probe</p>
        </div>
      </header>
      <section className="gate">
        <p className="lede">This demo is password protected. Enter the access password to continue.</p>
        {note ? <p className="notice">{note}</p> : null}
        <form className="gate-form" onSubmit={submit}>
          <input
            type="password"
            value={password}
            autoComplete="current-password"
            aria-label="Access password"
            placeholder="access password"
            disabled={busy}
            onChange={(event) => setPassword(event.target.value)}
          />
          <button type="submit" className="primary" disabled={busy || password === ""}>
            {busy ? "Checking" : "Continue"}
          </button>
        </form>
        {rejected ? <p className="error">That password was not accepted</p> : null}
        {status === "offline" ? (
          <p className="hint">
            The server is offline right now, so the password cannot be checked.{" "}
            <button type="button" className="link" onClick={onContinueOffline}>
              Continue without a password
            </button>{" "}
            to use the prebuilt examples.
          </p>
        ) : null}
      </section>
    </>
  );
}
