import { useState } from "react";
import { formatPlain } from "../colors";
import type { FitResponse } from "../types";
import { Sparkline } from "./Sparkline";

interface Props {
  fit: FitResponse;
  layer: number;
  onLayerChange: (layer: number) => void;
}

const PLATEAU = 0.01;

// The layer with the highest confound index among layers whose CV AUROC sits within
// PLATEAU of the best, i.e. the most concept-tracking layer that still separates the piles.
// The backend's accuracy pick ignores the first quarter of the layers, where directions read
// words and steering collapses; the plateau search starts at the same floor.
export function layerFloor(nLayers: number): number {
  return Math.floor(nLayers / 4);
}

export function bestConfoundLayer(fit: FitResponse): number {
  const auroc = fit.per_layer.cv_auroc;
  const floor = Math.min(layerFloor(fit.n_layers), Math.max(0, auroc.length - 1));
  const top = Math.max(...auroc.slice(floor));
  let best = fit.picked_layer;
  let bestIndex = Number.NEGATIVE_INFINITY;
  for (let i = floor; i < auroc.length; i += 1) {
    if (auroc[i] >= top - PLATEAU) {
      const value = fit.confound.index[i] ?? Number.NEGATIVE_INFINITY;
      if (value > bestIndex) {
        bestIndex = value;
        best = i;
      }
    }
  }
  return best;
}

// One sentence naming the layer in view relative to the two candidate defaults, shared by
// the layer strip and the monitor header so both disclose the same thing.
export function describeLayer(fit: FitResponse, layer: number): string {
  const best = bestConfoundLayer(fit);
  const picked = fit.picked_layer;
  if (layer === best) {
    return `viewing layer ${layer}, best confound among the plateau; accuracy pick ${picked}`;
  }
  if (layer === picked) {
    return `viewing layer ${layer}, the accuracy pick; best confound among the plateau ${best}`;
  }
  return `viewing layer ${layer}; best confound among the plateau ${best}; accuracy pick ${picked}`;
}

function pct(value: number | undefined): string {
  return value === undefined ? "n/a" : `${Math.round(value * 100)}%`;
}

function two(value: number | undefined): string {
  return value === undefined ? "n/a" : formatPlain(value, 2);
}

export function LayerStrip({ fit, layer, onLayerChange }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const auroc = fit.per_layer.cv_auroc;
  const accuracy = fit.per_layer.heldout_acc;
  const pooled = fit.per_layer.heldout_auroc ?? [];
  const matched = fit.per_layer.heldout_auroc_matched ?? [];
  const background = fit.per_layer.heldout_auroc_background ?? [];
  const index = fit.confound.index;
  const count = auroc.length;
  const shown = hover ?? layer;
  const best = bestConfoundLayer(fit);
  const floor = layerFloor(fit.n_layers);
  const stability = fit.stability ?? null;
  const low = Math.min(0.5, ...auroc, ...matched, ...background) - 0.02;
  const indexLow = Math.min(0, ...index);
  const indexHigh = Math.max(0, ...index);
  const pad = Math.max(0.1, (indexHigh - indexLow) * 0.1);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Layers</h2>
        <span className="badge">
          held-out accuracy {pct(accuracy[layer])} at layer {layer}
        </span>
      </div>
      <Sparkline
        values={auroc}
        extra={[
          { values: matched, className: "spark-extra" },
          { values: background, className: "spark-extra spark-extra-dashed" },
        ]}
        domain={[low, 1]}
        baseline={0.5}
        selected={layer}
        marked={fit.picked_layer}
        onHover={setHover}
        onSelect={onLayerChange}
        label="Cross-validated AUROC by layer, with matched-only and background-only diagnostics"
      />
      <div className="readout">
        layer {shown} · CV {two(auroc[shown])} · held-out {two(accuracy[shown])}, pooled {two(pooled[shown])} / matched {two(matched[shown])} /
        background {two(background[shown])}
        {shown === fit.picked_layer ? " · accuracy pick" : ""}
      </div>
      {stability ? (
        <div className="readout">
          stability: split-half {two(stability.split_half_cosine[shown])}, projected {two(stability.full_set_reliability[shown])}, split-half
          AUROC {two(stability.split_half_auroc[shown])}
        </div>
      ) : null}
      <h3 className="subhead">Confound index by layer</h3>
      <Sparkline
        values={index}
        domain={[indexLow - pad, indexHigh + pad]}
        baseline={0}
        selected={layer}
        marked={best}
        height={64}
        onHover={setHover}
        onSelect={onLayerChange}
        label="Confound index by layer"
      />
      <div className="readout readout-row">
        <span>
          confound index {two(index[shown])} at layer {shown}; best plateau layer {best} ({two(index[best])})
        </span>
        <span className={`slider-link${layer === best ? " is-hidden" : ""}`}>
          <button type="button" className="link" onClick={() => onLayerChange(best)}>
            jump to best confound layer ({best})
          </button>
        </span>
      </div>
      <details className="collapsible how">
        <summary>How to read this</summary>
        <p className="hint">
          Top chart: cross-validated AUROC of the shipped direction at each of the {count} layers. Bold line: the shipped direction, fit on
          matched plus background negatives; thin lines: directions fit on matched negatives only (solid) and on background plus neutral only
          (dashed), all scored on the same held-out set. The accuracy pick (ring) is the first layer within 0.01 of the best, searched from a
          quarter of the way in (layer {floor} on), because earlier layers read words and collapse under steering. Accuracy saturates early on
          many concepts, so the default view is the plateau layer with the best confound index, searched from the same floor ({describeLayer(fit, layer)}).
          Click or drag either chart to read every panel and heatmap at another layer; nothing is refetched.
        </p>
        <p className="hint">
          Bottom chart: implicit positives minus decoys at each layer. It can change sign with depth: a direction that reads the words in early
          layers can read the concept deeper in. The ring marks the best index among layers from a quarter of the way in that sit within 0.01 of
          the top CV AUROC there, which is the default layer.
        </p>
        {stability ? (
          <p className="hint">
            Stability is the cosine between directions fit on two random halves of the training pools, projected to the full set ({stability.n_splits}{" "}
            splits); below about 0.7 projected, more examples would still change the direction.
          </p>
        ) : null}
      </details>
    </section>
  );
}
