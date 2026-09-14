// The access password lives under its own localStorage key, apart from the session, and is
// sent as the X-Probe-Key header on every API call.
const STORAGE_KEY = "probe-bench.key.v1";

export function getStoredKey(): string | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value && value !== "" ? value : null;
  } catch {
    return null;
  }
}

export function setStoredKey(key: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, key);
  } catch {
    // Storage unavailable: the key lives in memory for this page only.
  }
}

export function clearStoredKey(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored.
  }
}
