import { useEffect, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import { formatZ, zToBackground } from "../colors";

export interface TokenFocus {
  index: number;
  pinned: boolean;
  source?: string; // the heatmap the focus came from, when two share one
}

// The other probe's values for the same tokens; ownFirst says whether this heatmap's value is
// read first in "a vs b" (the current probe always comes first).
export interface TokenPair {
  z: number[][];
  ownFirst: boolean;
}

interface Props {
  tokens: string[];
  z: number[][];
  layer: number;
  dim?: boolean;
  focus?: TokenFocus | null;
  onFocusChange?: (focus: TokenFocus | null) => void;
  id?: string;
  pair?: TokenPair;
}

const TIP_GAP = 5;

function valueAt(z: number[][], index: number, layer: number): number {
  const row = z[index];
  return row !== undefined && layer < row.length ? row[layer] : 0;
}

// Whitespace-only tokens still need a visible name in the readouts.
function tokenLabel(token: string): string {
  const visible = token.replace(/\n/g, "⏎").trim();
  if (visible !== "") {
    return visible;
  }
  return token.includes("\n") ? "⏎" : "␣";
}

function describe(tokens: string[], z: number[][], layer: number, index: number, pair?: TokenPair): string {
  const own = formatZ(valueAt(z, index, layer), 2);
  if (!pair) {
    return `${tokenLabel(tokens[index] ?? "")}  ${own} token units at layer ${layer}`;
  }
  const other = formatZ(valueAt(pair.z, index, layer), 2);
  const [first, second] = pair.ownFirst ? [own, other] : [other, own];
  return `${tokenLabel(tokens[index] ?? "")}  ${first} vs ${second} token units at layer ${layer}`;
}

// Only tokens with a colored span can be focused: partial byte tokens arrive as "".
function isVisible(tokens: string[], index: number): boolean {
  return index < tokens.length && tokens[index] !== "";
}

interface Placement {
  tip: CSSProperties;
  ring: CSSProperties;
}

// A token span may break at its leading space, leaving the space on one line and the word
// on the next; the tooltip and the ring anchor to the last fragment, the one holding the
// word. Both are position: fixed from viewport rectangles, so the scrolling messages list
// cannot clip them: the tooltip sits above the fragment, or below it when the fragment is on
// the first line of its turn (so the turn head stays readable) or at the top of the list,
// hanging from the right edge in the right part of the column. A span the list has
// scrolled out of view gets neither.
function place(span: HTMLElement): Placement | null {
  const rects = [...span.getClientRects()];
  const fragments = rects.filter((fragment) => fragment.width > 0);
  const rect = fragments[fragments.length - 1] ?? rects[rects.length - 1] ?? span.getBoundingClientRect();
  const list = span.closest(".messages")?.getBoundingClientRect();
  if (list && (rect.bottom < list.top || rect.top > list.bottom)) {
    return null;
  }
  const body = span.closest(".turn-body")?.getBoundingClientRect();
  const left = list?.left ?? 0;
  const width = list?.width ?? window.innerWidth;
  const top = list?.top ?? 0;
  const tip: CSSProperties = { position: "fixed" };
  if (rect.left + rect.width / 2 > left + width * 0.6) {
    tip.right = window.innerWidth - rect.right;
  } else {
    tip.left = rect.left;
  }
  const firstLine = body !== undefined && rect.top - body.top < rect.height / 2;
  if (firstLine || rect.top - top < 28) {
    tip.top = rect.bottom + TIP_GAP;
  } else {
    tip.bottom = window.innerHeight - rect.top + TIP_GAP;
  }
  const ring: CSSProperties = { position: "fixed", left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  return { tip, ring };
}

// Focus state for one turn's tokens: hover follows the mouse, a click pins the token so the
// readout stays (and works on touch); Escape clears a pin.
export function useTokenFocus(): [TokenFocus | null, (focus: TokenFocus | null) => void] {
  const [focus, setFocus] = useState<TokenFocus | null>(null);
  useEffect(() => {
    if (!focus?.pinned) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setFocus(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focus]);
  return [focus, setFocus];
}

export function TokenReadout({
  tokens,
  z,
  layer,
  focus,
  pair,
}: {
  tokens: string[];
  z: number[][];
  layer: number;
  focus: TokenFocus | null;
  pair?: TokenPair;
}) {
  if (!focus || !isVisible(tokens, focus.index)) {
    return null;
  }
  return (
    <span className="tok-readout">
      {focus.pinned ? "pinned  " : ""}
      {describe(tokens, z, layer, focus.index, pair)}
      {focus.pinned ? "  (Esc clears)" : ""}
    </span>
  );
}

export function TokenHeatmap({ tokens, z, layer, dim = false, focus = null, onFocusChange, id, pair }: Props) {
  const anchorRef = useRef<HTMLElement | null>(null);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const active = focus && isVisible(tokens, focus.index) ? focus.index : null;
  // Two heatmaps can share one focus; only the one the pointer is on draws the ring and tip.
  const owns = focus?.source === id;

  // Keep the tooltip and ring on their fragment while anything scrolls, resizes, or
  // reflows: while a token is focused, the span is re-measured every frame and the styles
  // updated only when they change.
  useEffect(() => {
    if (active === null) {
      return;
    }
    let frame = 0;
    let last = "";
    const tick = () => {
      if (anchorRef.current) {
        const next = place(anchorRef.current);
        const key = JSON.stringify(next);
        if (key !== last) {
          last = key;
          setPlacement(next);
        }
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [active]);

  const anchor = (event: MouseEvent<HTMLElement>) => {
    anchorRef.current = event.currentTarget;
    setPlacement(place(event.currentTarget));
  };

  const enter = (index: number, event: MouseEvent<HTMLElement>) => {
    if (!onFocusChange || focus?.pinned) {
      return;
    }
    anchor(event);
    onFocusChange({ index, pinned: false, source: id });
  };

  const leave = (index: number) => {
    if (!onFocusChange || !focus || focus.pinned || focus.index !== index) {
      return;
    }
    onFocusChange(null);
  };

  const click = (index: number, event: MouseEvent<HTMLElement>) => {
    if (!onFocusChange) {
      return;
    }
    if (focus?.pinned && focus.index === index) {
      onFocusChange({ index, pinned: false, source: id });
      return;
    }
    anchor(event);
    onFocusChange({ index, pinned: true, source: id });
  };

  // The ring and tooltip live outside the token spans so their fragments stay untouched.
  return (
    <>
      <span className={`tokens${dim ? " tokens-dim" : ""}`}>
        {tokens.map((token, i) => {
          if (token === "") {
            return null;
          }
          return (
            <span
              key={i}
              className="tok"
              style={{ backgroundColor: zToBackground(valueAt(z, i, layer)) }}
              onMouseEnter={(event) => enter(i, event)}
              onMouseLeave={() => leave(i)}
              onClick={(event) => click(i, event)}
            >
              {token}
            </span>
          );
        })}
      </span>
      {active !== null && placement && owns ? (
        <>
          <span className={`tok-ring${focus?.pinned ? " tok-ring-pinned" : ""}`} style={placement.ring} />
          <span className="tok-tip" role="tooltip" style={placement.tip}>
            {describe(tokens, z, layer, active, pair)}
          </span>
        </>
      ) : null}
    </>
  );
}
