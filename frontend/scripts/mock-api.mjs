// Serves the repo fixtures as the Probe Bench API so the frontend can run with the pod down.
// Usage: node scripts/mock-api.mjs [port]   then point public/config.json at http://127.0.0.1:<port>
// Fits, scores, axis pairs, and deflation come from fixtures/; concept generation and chat answer 400.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "..", "..", "fixtures");
const example = JSON.parse(readFileSync(join(fixtures, "example.json"), "utf8"));
const axisExample = JSON.parse(readFileSync(join(fixtures, "axis-example.json"), "utf8"));
const mineExample = JSON.parse(readFileSync(join(fixtures, "mine-example.json"), "utf8"));
const port = Number(process.argv[2] ?? 8787);
// Every route except GET /health and GET /auth needs this header value.
const ACCESS_KEY = process.env.MOCK_KEY ?? "mock-key";
const probes = new Map([
  [example.fit.probe_id, example.fit],
  [axisExample.deflate.fit.probe_id, axisExample.deflate.fit],
]);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Accept, X-Probe-Key",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function hash(text) {
  let h = 2166136261;
  for (const ch of text) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

// A whitespace split stands in for the tokenizer; values are deterministic per probe, token,
// and eight-layer band, so two probes differ and the layer slider has something to re-read.
function scoreTurn(role, text, probeId) {
  const layers = example.fit.n_layers;
  const tokens = text.match(/\s*\S+/g) ?? [];
  const z = tokens.map((token) =>
    Array.from({ length: layers }, (_, layer) => ((hash(`${probeId}|${token.trim().toLowerCase()}|${layer >> 3}`) % 2000) / 1000 - 1) * 1.5),
  );
  const seq_z = Array.from({ length: layers }, (_, layer) => (z.length > 0 ? z.reduce((sum, row) => sum + row[layer], 0) / z.length : 0));
  return { role, text, tokens, z, seq_z };
}

function json(res, status, body) {
  res.writeHead(status, { ...CORS, "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve(null);
      }
    });
  });
}

async function sse(res, events) {
  res.writeHead(200, { ...CORS, "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  for (const [delay, event] of events) {
    await sleep(delay);
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
  res.end();
}

createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
    res.end();
    return;
  }
  if (req.method === "GET" && url.pathname === "/health") {
    json(res, 200, { model_loaded: true, loading: false, model_id: "mock (fixtures)", n_layers: example.fit.n_layers });
    return;
  }
  const authorized = req.headers["x-probe-key"] === ACCESS_KEY;
  if (req.method === "GET" && url.pathname === "/auth") {
    if (authorized) {
      json(res, 200, { ok: true });
    } else {
      json(res, 401, { error: "password required" });
    }
    return;
  }
  if (!authorized) {
    json(res, 401, { error: "password required" });
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/probe/")) {
    const id = decodeURIComponent(url.pathname.slice("/probe/".length));
    const fit = probes.get(id);
    if (!fit) {
      json(res, 404, { error: `probe ${id} is not on this server` });
      return;
    }
    json(res, 200, { ...fit, contrast_set: example.contrast_set });
    return;
  }
  const body = req.method === "POST" ? await readBody(req) : {};
  if (body === null) {
    json(res, 400, { error: "the request body is not JSON" });
    return;
  }
  switch (`${req.method} ${url.pathname}`) {
    case "POST /fit": {
      await sleep(800);
      json(res, 200, example.fit);
      return;
    }
    case "POST /score": {
      // The request's own text, split on whitespace and colored per probe id, so edits,
      // comparisons, and the layer slider all read back what was sent.
      const messages = Array.isArray(body.messages) ? body.messages : [];
      const turns = [];
      if (messages.length > 0 && typeof body.system_prompt === "string" && body.system_prompt.trim() !== "") {
        turns.push(scoreTurn("system", body.system_prompt, String(body.probe_id)));
      }
      for (const message of messages) {
        turns.push(scoreTurn(message.role, String(message.content ?? ""), String(body.probe_id)));
      }
      json(res, 200, { turns, conversation_tokens: turns.reduce((n, turn) => n + turn.tokens.length, 0) });
      return;
    }
    case "POST /axis": {
      const axis = body.contrast_set?.confound_axis;
      if (typeof axis !== "string" || axis === "") {
        json(res, 400, { error: "contrast_set.confound_axis names the axis to fit" });
        return;
      }
      await sse(res, [
        [0, { type: "status", text: `writing ${axis} pairs (this can take a minute)` }],
        [1500, { type: "status", text: "checking that each pair holds the concept fixed (20 s)" }],
        [1500, { type: "axis", ...axisExample.axis, axis_set: { ...axisExample.axis.axis_set, axis } }],
      ]);
      return;
    }
    case "POST /axis.json": {
      json(res, 200, axisExample.axis);
      return;
    }
    case "POST /deflate": {
      if (typeof body.probe_id !== "string" || typeof body.axis_set?.axis !== "string") {
        json(res, 400, { error: "probe_id and axis_set are required" });
        return;
      }
      if (!probes.has(body.probe_id)) {
        json(res, 404, { error: `probe ${body.probe_id} is not on this server` });
        return;
      }
      await sleep(1200);
      const axis = body.axis_set.axis;
      const requested = Math.min(4, Math.max(1, Math.round(Number(body.rounds ?? 1)) || 1));
      // Three or more requested rounds stop one short, the way the stop rule does on the pod.
      const rounds = requested >= 3 ? requested - 1 : requested;
      // Each fixture round walks the before arrays toward the after arrays; the last round lands on them.
      const base = axisExample.deflate;
      const step = (before, after, k) => before.map((value, i) => value + (after[i] - value) * (k / rounds));
      const round_history = Array.from({ length: rounds }, (_, i) => {
        const round = i + 1;
        let note = round === 1 ? "axis: mean difference of the pairs" : "axis: top principal direction of the deflated differences";
        if (round === rounds && rounds < requested) {
          note += "; stopped: another round would lower the held-out AUROC at the picked layer";
        }
        return {
          round,
          note,
          axis_cv_auroc: step(base.axis_cv_auroc_before, base.axis_cv_auroc_after, round),
          heldout_auroc: step(base.heldout_auroc_before, base.heldout_auroc_after, round),
          confound_index: step(base.confound_index_before, base.confound_index_after, round),
        };
      });
      json(res, 200, { ...base, axis, rounds_applied: rounds, round_history, fit: { ...base.fit, deflated_axis: axis } });
      return;
    }
    case "POST /misfires": {
      if (typeof body.probe_id !== "string") {
        json(res, 400, { error: "probe_id is required" });
        return;
      }
      if (!probes.has(body.probe_id)) {
        json(res, 404, { error: `probe ${body.probe_id} is not on this server` });
        return;
      }
      await sleep(600);
      json(res, 200, { ...mineExample.misfires, layer: typeof body.layer === "number" ? body.layer : mineExample.misfires.layer });
      return;
    }
    case "POST /mine": {
      if (!Array.isArray(body.seeds) || body.seeds.length === 0) {
        json(res, 400, { error: "seeds must be a non-empty list" });
        return;
      }
      await sse(res, [
        [0, { type: "status", text: `writing decoys around ${body.seeds.length} seeds (this can take a minute)` }],
        [1500, { type: "status", text: "checking that each decoy keeps the concept absent (20 s)" }],
        [1500, { type: "mined", ...mineExample.mine }],
      ]);
      return;
    }
    case "POST /mine.json": {
      json(res, 200, mineExample.mine);
      return;
    }
    case "POST /concept": {
      json(res, 400, { error: "concept generation is not mocked; pick a prebuilt example" });
      return;
    }
    case "POST /chat": {
      json(res, 400, { error: "chat is not mocked" });
      return;
    }
    default: {
      json(res, 404, { error: `no route for ${req.method} ${url.pathname}` });
    }
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`mock api on http://127.0.0.1:${port}`);
});
