import type { MouseEvent } from "react";
import { zToSolid } from "../colors";

const W = 600;
const PAD = 8;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export interface ExtraSeries {
  values: number[];
  className: string;
}

export interface SparklineProps {
  values: number[];
  domain: [number, number];
  label: string;
  height?: number;
  baseline?: number;
  selected?: number | null;
  marked?: number | null;
  markers?: boolean;
  extra?: ExtraSeries[];
  onHover?: (index: number | null) => void;
  onSelect?: (index: number) => void;
}

// Hand-rolled SVG line. The viewBox is fixed and the element scales uniformly to its
// container, so hit-testing maps the pointer's x fraction straight to an index.
export function Sparkline({
  values,
  domain,
  label,
  height = 72,
  baseline,
  selected = null,
  marked = null,
  markers = false,
  extra = [],
  onHover,
  onSelect,
}: SparklineProps) {
  const n = values.length;
  const x = (i: number) => (n <= 1 ? W / 2 : PAD + (i / (n - 1)) * (W - 2 * PAD));
  const y = (v: number) => {
    const [lo, hi] = domain;
    const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
    return PAD + (1 - clamp(t, 0, 1)) * (height - 2 * PAD);
  };
  const pathFor = (series: number[]) =>
    series
      .slice(0, n)
      .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`)
      .join(" ");

  const indexAt = (event: MouseEvent<SVGSVGElement>): number | null => {
    if (n === 0) {
      return null;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) {
      return null;
    }
    const px = ((event.clientX - rect.left) / rect.width) * W;
    return clamp(Math.round(((px - PAD) / (W - 2 * PAD)) * (n - 1)), 0, n - 1);
  };

  const interactive = Boolean(onHover || onSelect);
  return (
    <svg
      className={`sparkline${interactive ? " sparkline-interactive" : ""}`}
      viewBox={`0 0 ${W} ${height}`}
      role="img"
      aria-label={label}
      onMouseMove={onHover ? (event) => onHover(indexAt(event)) : undefined}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
      onClick={
        onSelect
          ? (event) => {
              const index = indexAt(event);
              if (index !== null) {
                onSelect(index);
              }
            }
          : undefined
      }
    >
      {baseline !== undefined ? <line className="spark-baseline" x1={PAD} x2={W - PAD} y1={y(baseline)} y2={y(baseline)} /> : null}
      {selected !== null && selected >= 0 && selected < n ? (
        <line className="spark-selected" x1={x(selected)} x2={x(selected)} y1={2} y2={height - 2} />
      ) : null}
      {extra.map((series, i) => (
        <path key={i} className={series.className} d={pathFor(series.values)} />
      ))}
      {n > 0 ? <path className="spark-line" d={pathFor(values)} /> : null}
      {marked !== null && marked >= 0 && marked < n ? <circle className="spark-marked" cx={x(marked)} cy={y(values[marked])} r={5} /> : null}
      {markers ? values.map((v, i) => <circle key={i} className="spark-marker" cx={x(i)} cy={y(v)} r={4} fill={zToSolid(v)} />) : null}
    </svg>
  );
}
