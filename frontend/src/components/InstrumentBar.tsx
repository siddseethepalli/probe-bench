import type { ReactNode } from "react";
import { formatPlain } from "../colors";
import type { FitResponse } from "../types";
import { bestConfoundLayer } from "./LayerStrip";

interface Props {
  fit: FitResponse;
  layer: number;
  concept: string;
  minedCount: number;
  onLayerChange: (layer: number) => void;
  probeActions?: ReactNode;
}

// The one place the layer is chosen: a thin bar that sticks to the top of the viewport once a
// fit is loaded, so both columns follow the same layer wherever the page is scrolled.
export function InstrumentBar({ fit, layer, concept, minedCount, onLayerChange, probeActions }: Props) {
  const count = fit.n_layers;
  const best = bestConfoundLayer(fit);
  const verdict = fit.verdict_by_layer[layer] ?? "mixed";
  const index = fit.confound.index[layer] ?? 0;
  const stability = fit.stability ?? null;
  return (
    <div className="instrument-bar" role="region" aria-label="Instrument bar">
      <span className="bar-concept">
        <span className="bar-name" title={concept}>
          {concept}
        </span>
        {fit.deflated_axis ? <span className="badge">deflated: {fit.deflated_axis}</span> : null}
        {minedCount > 0 ? <span className="badge">mined: {minedCount}</span> : null}
      </span>
      <span className="badge bar-verdict" title={`Verdict and confound index at layer ${layer}`}>
        {verdict} · index {formatPlain(index, 2)}
      </span>
      <div className="slider-row bar-slider" data-tour="layer">
        <label htmlFor="layer-slider">layer</label>
        <input
          id="layer-slider"
          type="range"
          min={0}
          max={Math.max(0, count - 1)}
          step={1}
          value={layer}
          onChange={(event) => onLayerChange(Number(event.target.value))}
        />
        <output htmlFor="layer-slider">{layer}</output>
        <span className={`slider-link${layer === fit.picked_layer ? " is-hidden" : ""}`}>
          <button type="button" className="link" onClick={() => onLayerChange(fit.picked_layer)}>
            accuracy pick ({fit.picked_layer})
          </button>
        </span>
        <span className={`slider-link${layer === best ? " is-hidden" : ""}`}>
          <button type="button" className="link" onClick={() => onLayerChange(best)}>
            best confound ({best})
          </button>
        </span>
      </div>
      {stability ? (
        <span className="chip bar-stability" title="Direction stability, split-half cosine projected to the full set">
          stability {formatPlain(stability.full_set_reliability[layer] ?? 0, 2)}
        </span>
      ) : null}
      {probeActions}
    </div>
  );
}
