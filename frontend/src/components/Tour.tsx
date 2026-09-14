import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { findTourTarget, TOUR_STEPS } from "../tour";

interface Props {
  onClose: () => void;
}

const PAD = 6;
const GAP = 12;
const MARGIN = 16;
// The least of a tall target the spotlight keeps when the card has to take the rest of the screen.
const MIN_SPOT = 120;

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

function focusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>("button:not(:disabled), [href], input, textarea, select, [tabindex]:not([tabindex='-1'])"));
}

// Steps whose target is not on the page are skipped in the direction of travel.
function nextAvailable(from: number, direction: 1 | -1): number | null {
  for (let i = from + direction; i >= 0 && i < TOUR_STEPS.length; i += direction) {
    if (findTourTarget(TOUR_STEPS[i].target)) {
      return i;
    }
  }
  return null;
}

// Space the sticky instrument bar takes at the top of the viewport, so a target is never
// scrolled underneath it.
function guardTop(): number {
  const bar = document.querySelector<HTMLElement>(".instrument-bar");
  return (bar ? bar.getBoundingClientRect().height : 0) + MARGIN;
}

function scrollToTarget(target: HTMLElement): void {
  const rect = target.getBoundingClientRect();
  const viewHeight = window.innerHeight;
  const guard = guardTop();
  if (rect.top >= guard && rect.bottom <= viewHeight - MARGIN) {
    return;
  }
  const room = viewHeight - guard - MARGIN;
  const delta = rect.height > room ? rect.top - guard : rect.top - guard - (room - rect.height) / 2;
  window.scrollBy({ top: delta, behavior: "auto" });
}

// Lays the spotlight over the target and the card beside it. Runs every frame while the tour is
// open, so the page can be scrolled and read underneath; everything is written straight to the
// two elements to keep React out of the per-frame path.
function layout(target: HTMLElement | null, spot: HTMLElement, card: HTMLElement): void {
  const viewWidth = window.innerWidth;
  const viewHeight = window.innerHeight;
  const cardWidth = card.offsetWidth;
  const cardHeight = card.offsetHeight;
  if (!target) {
    spot.style.display = "none";
    card.style.left = `${Math.max(MARGIN, (viewWidth - cardWidth) / 2)}px`;
    card.style.top = `${Math.max(MARGIN, (viewHeight - cardHeight) / 2)}px`;
    return;
  }
  spot.style.display = "";
  const rect = target.getBoundingClientRect();
  // The sticky bar stays dimmed when a scrolled target runs underneath it.
  const bar = document.querySelector<HTMLElement>(".instrument-bar");
  const barBottom = bar && !bar.contains(target) ? bar.getBoundingClientRect().bottom : 0;
  const top = Math.max(rect.top - PAD, Math.min(barBottom, rect.bottom - PAD));
  let bottom = rect.bottom + PAD;
  const left = rect.left - PAD;
  const right = rect.right + PAD;
  const fitsBelow = viewHeight - bottom >= cardHeight + GAP + MARGIN;
  const fitsAbove = top >= cardHeight + GAP + MARGIN;
  const fitsRight = viewWidth - right >= cardWidth + GAP + MARGIN;
  const fitsLeft = left >= cardWidth + GAP + MARGIN;
  let cardTop: number;
  let cardLeft: number | null = null;
  if (fitsBelow) {
    cardTop = bottom + GAP;
  } else if (fitsAbove) {
    cardTop = top - GAP - cardHeight;
  } else if (fitsRight) {
    cardLeft = right + GAP;
    cardTop = clamp(top, MARGIN, viewHeight - cardHeight - MARGIN);
  } else if (fitsLeft) {
    cardLeft = left - GAP - cardWidth;
    cardTop = clamp(top, MARGIN, viewHeight - cardHeight - MARGIN);
  } else {
    // A target taller than the room left: light its top and cut the spotlight where the card starts.
    const cut = viewHeight - cardHeight - GAP - MARGIN;
    if (cut - Math.max(top, 0) >= MIN_SPOT) {
      bottom = cut;
      cardTop = cut + GAP;
    } else {
      cardTop = viewHeight - cardHeight - MARGIN;
    }
  }
  if (cardLeft === null) {
    cardLeft = clamp(left, MARGIN, Math.max(MARGIN, viewWidth - cardWidth - MARGIN));
  }
  spot.style.top = `${top}px`;
  spot.style.left = `${left}px`;
  spot.style.width = `${right - left}px`;
  spot.style.height = `${bottom - top}px`;
  card.style.top = `${cardTop}px`;
  card.style.left = `${cardLeft}px`;
}

// Spotlight tour over the live page: one card, six targets, the rest of the page dimmed but
// still readable and scrollable. Focus stays inside the card; Escape skips.
export function Tour({ onClose }: Props) {
  const [index, setIndex] = useState(() => nextAvailable(-1, 1) ?? 0);
  const spotRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const step = TOUR_STEPS[index];
  const hasPrev = nextAvailable(index, -1) !== null;
  const hasNext = nextAvailable(index, 1) !== null;

  const next = useCallback(() => {
    const target = nextAvailable(index, 1);
    if (target === null) {
      onClose();
    } else {
      setIndex(target);
    }
  }, [index, onClose]);

  const back = useCallback(() => {
    const target = nextAvailable(index, -1);
    if (target !== null) {
      setIndex(target);
    }
  }, [index]);

  // Focus moves into the card while the tour is open and returns where it was afterwards.
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    return () => {
      if (previous && previous.isConnected) {
        previous.focus({ preventScroll: true });
      }
    };
  }, []);

  useLayoutEffect(() => {
    const target = findTourTarget(step.target);
    if (target && step.scroll) {
      scrollToTarget(target);
    }
    const spot = spotRef.current;
    const card = cardRef.current;
    if (spot && card) {
      layout(target, spot, card);
      if (!card.contains(document.activeElement)) {
        card.focus({ preventScroll: true });
      }
    }
  }, [step]);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const spot = spotRef.current;
      const card = cardRef.current;
      if (spot && card) {
        layout(findTourTarget(step.target), spot, card);
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [step]);

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const onCardKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      next();
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      back();
      return;
    }
    if (event.key !== "Tab") {
      return;
    }
    const card = cardRef.current;
    if (!card) {
      return;
    }
    const items = focusables(card);
    if (items.length === 0) {
      event.preventDefault();
      return;
    }
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || active === card)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <>
      <div ref={spotRef} className="tour-spot" aria-hidden="true" />
      <div ref={cardRef} className="tour-card" role="dialog" aria-labelledby="tour-title" aria-describedby="tour-text" tabIndex={-1} onKeyDown={onCardKey}>
        <div className="tour-head">
          <span id="tour-title" className="tour-title">
            {step.title}
          </span>
          <span className="muted">
            {index + 1} of {TOUR_STEPS.length}
          </span>
        </div>
        <p id="tour-text" className="tour-text">
          {step.text}
        </p>
        <div className="tour-actions">
          <button type="button" onClick={back} disabled={!hasPrev}>
            Back
          </button>
          <button type="button" className="primary" onClick={next}>
            {hasNext ? "Next" : "Done"}
          </button>
          <button type="button" className="link" onClick={onClose}>
            Skip
          </button>
        </div>
      </div>
    </>
  );
}
