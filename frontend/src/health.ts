import { useCallback, useEffect, useState } from "react";
import { getHealth } from "./api";
import type { Health } from "./types";

export type Status = "checking" | "online" | "offline";

const POLL_MS = 10_000;
const TIMEOUT_MS = 6_000;

// Polls GET /health every ten seconds. "online" means the server answered and the
// model is loaded; anything else is "offline" and the page falls back to static data.
export function useHealth(apiBase: string | null): {
  status: Status;
  health: Health | null;
  retryIn: number;
  checkNow: () => void;
} {
  const [status, setStatus] = useState<Status>("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [nextAt, setNextAt] = useState<number | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (apiBase === null) {
      return;
    }
    if (apiBase === "") {
      setStatus("offline");
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const run = async () => {
      try {
        const next = await getHealth(apiBase, TIMEOUT_MS);
        if (cancelled) {
          return;
        }
        setHealth(next);
        setStatus(next.model_loaded ? "online" : "offline");
      } catch {
        if (cancelled) {
          return;
        }
        setStatus("offline");
      }
      if (!cancelled) {
        setNextAt(Date.now() + POLL_MS);
        setNow(Date.now());
        timer = window.setTimeout(run, POLL_MS);
      }
    };
    void run();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [apiBase, nonce]);

  useEffect(() => {
    if (status !== "offline") {
      return;
    }
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [status]);

  const checkNow = useCallback(() => {
    setStatus("checking");
    setNonce((n) => n + 1);
  }, []);

  const retryIn = nextAt === null ? 0 : Math.max(0, Math.ceil((nextAt - now) / 1000));
  return { status, health, retryIn, checkNow };
}
