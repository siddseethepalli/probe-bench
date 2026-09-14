// The first-visit tour runs once per browser; the flag lives beside the session and the
// access password under its own key so clearing either leaves the other alone.
const STORAGE_KEY = "probe-bench.tour.v1";

export function hasSeenTour(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) !== null;
  } catch {
    return true;
  }
}

export function markTourSeen(): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, "seen");
  } catch {
    // Storage unavailable: the tour shows again next time.
  }
}

export interface TourStep {
  target: string;
  title: string;
  text: string;
  // The layer slider lives in the sticky bar, which is on screen wherever the page is scrolled.
  scroll: boolean;
}

// Each target is an element carrying data-tour="<target>"; a step whose target is not on the
// page (an empty page, or a panel that has not loaded) is skipped.
export const TOUR_STEPS: TourStep[] = [
  {
    target: "concept",
    title: "Concept",
    text: "Type a concept, or pick a prebuilt one. The model writes the two piles: examples with the concept and examples without it.",
    scroll: true,
  },
  {
    target: "verdict",
    title: "Verdict",
    text: "Implicit positives carry the concept without its words, decoys carry the words without the concept. If they score alike, the direction is a word detector.",
    scroll: true,
  },
  {
    target: "fixes",
    title: "Fixes",
    text: "Three ways to fix a set. Add harder negatives, remove the tone it rides, or ask the probe where it misfires.",
    scroll: true,
  },
  {
    target: "layer",
    title: "Layer",
    text: "Every panel and heatmap re-reads at any layer. The default is the best confound layer on the accuracy plateau.",
    scroll: false,
  },
  {
    target: "monitor",
    title: "Monitor",
    text: "Chat under a system prompt you can edit; every token is colored by the probe. Hover a token for its value, and switch probes to compare.",
    scroll: true,
  },
  {
    target: "data",
    title: "Data",
    text: "This is the whole probe. Edit any example and refit.",
    scroll: true,
  },
];

export function findTourTarget(target: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[data-tour="${target}"]`);
}
