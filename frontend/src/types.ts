// Mirrors backend/contract.py exactly. Change both or neither.

export type Group = "explicit" | "implicit" | "decoys" | "neutral";
export type Role = "system" | "user" | "assistant";
export type Verdict = "tracks the concept" | "mixed" | "tracks the words" | "not tested";

export interface Example {
  text: string;
  context: string | null;
}

export interface ContrastSet {
  concept: string;
  is_response_property: boolean;
  keywords: string[];
  train_pos: Example[];
  train_neg: Example[];
  heldout_pos: Example[];
  heldout_neg: Example[];
  implicit_pos: Example[];
  decoys: Example[]; // first 6 feed the fix button, last 6 stay held out for the panel
  neutral: Example[];
  confound_axis: string | null; // the tone or style axis the stress sets decouple, named by the generator
  background: Example[]; // 24 same-format examples unrelated to the concept
  system_prompts: string[]; // [raise, lower]
  demo_user_message: string;
}

export interface ConceptResponse {
  contrast_set: ContrastSet;
  dropped: Record<string, number>; // per list, shortfall against the contract size after validation
  cached: boolean;
}

export type ConceptEvent =
  | { type: "status"; text: string }
  | ({ type: "concept" } & ConceptResponse)
  | { type: "error"; error: string };

export interface PerLayer {
  cv_auroc: number[]; // shipped direction, cross-validated on its own pool
  heldout_acc: number[]; // shipped direction on the matched held-out set
  heldout_auroc: number[]; // shipped direction on the matched held-out set
  heldout_auroc_matched: number[]; // direction fit on matched negatives only, same held-out set
  heldout_auroc_background: number[]; // direction fit on background + neutral only, same held-out set
}

export interface GroupMeans {
  explicit: number[];
  implicit: number[];
  decoys: number[];
  neutral: number[];
}

export interface Confound {
  index: number[];
  means: GroupMeans;
}

export interface StressExample {
  text: string;
  context: string | null;
  group: Group;
  seq_z: number[];
}

export interface FitResponse {
  probe_id: string;
  concept: string;
  model_id: string;
  n_layers: number;
  picked_layer: number;
  per_layer: PerLayer;
  confound: Confound;
  verdict_by_layer: Verdict[];
  stress_examples: StressExample[];
  deflated_axis: string | null; // set when a confound axis was projected out
  stability: Stability | null;
}

export interface Stability {
  split_half_cosine: number[]; // mean cosine between two half-directions
  full_set_reliability: number[]; // Spearman-Brown projection to the full set
  split_half_auroc: number[]; // direction from one half scored on the other
  n_splits: number;
}

export interface AxisPair {
  high: string;
  low: string;
  context: string | null;
}

export interface AxisSet {
  axis: string;
  absent_pairs: AxisPair[];
  present_pairs: AxisPair[];
}

export interface AxisResponse {
  axis_set: AxisSet;
  dropped: Record<string, number>;
  cached: boolean;
}

export type AxisEvent =
  | { type: "status"; text: string }
  | ({ type: "axis" } & AxisResponse)
  | { type: "error"; error: string };

export interface DeflateRequest {
  probe_id: string;
  axis_set: AxisSet;
  rounds?: number; // 1 to 4, default 1
}

export interface DeflateRound {
  round: number;
  note?: string | null;
  axis_cv_auroc: number[];
  heldout_auroc: number[];
  confound_index: number[];
}

export interface DeflateResponse {
  fit: FitResponse;
  axis: string;
  axis_cv_auroc_before: number[]; // leave-two-out separability of high vs low along the axis
  axis_cv_auroc_after: number[]; // the same after deflation; near 0.5 means the axis is gone
  heldout_auroc_before: number[];
  heldout_auroc_after: number[];
  confound_index_before: number[];
  confound_index_after: number[];
  rounds_applied: number;
  round_history: DeflateRound[];
}

export interface ProbeInfo extends FitResponse {
  contrast_set: ContrastSet;
}

export interface MisfiresRequest {
  probe_id: string;
  layer: number | null;
  top_k: number;
}

export interface Misfire {
  token: string;
  z: number;
  text: string;
  group: "matched" | "background" | "neutral" | "decoys"; // decoys = the first six, never the held-out panel
}

export interface MisfiresResponse {
  layer: number;
  misfires: Misfire[];
  seeds: string[];
}

export interface MineRequest {
  concept: string;
  contrast_set: ContrastSet;
  seeds: string[];
}

export interface MineResponse {
  decoys: Example[];
  dropped: Record<string, number>;
  cached: boolean;
}

export type MineEvent =
  | { type: "status"; text: string }
  | ({ type: "mined" } & MineResponse)
  | { type: "error"; error: string };

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  probe_id: string;
  system_prompt: string;
  messages: Message[];
  alpha: number;
  layer: number | null;
  max_new_tokens: number;
  temperature: number;
  seed: number;
}

export interface ScoredTurn {
  role: Role;
  text: string;
  tokens: string[];
  z: number[][]; // [T][L]
  seq_z: number[]; // [L]
}

export interface ChatScored {
  reply: string;
  turns: ScoredTurn[]; // every turn, system first when non-empty
  hit_length_cap: boolean;
  conversation_tokens: number;
  flags: string[];
}

export type ChatEvent =
  | { type: "token"; text: string }
  | ({ type: "scored" } & ChatScored)
  | { type: "error"; error: string };

export interface ScoreRequest {
  probe_id: string;
  system_prompt: string;
  messages: Message[];
}

export interface ScoreResponse {
  turns: ScoredTurn[];
  conversation_tokens: number;
}

export interface Health {
  model_loaded: boolean;
  loading: boolean;
  model_id: string;
  n_layers: number;
}

export interface StaticConversation {
  label: string;
  system_prompt: string;
  user_message: string;
  scored: ChatScored;
}

export interface StaticExample {
  slug: string;
  title: string;
  blurb: string;
  contrast_set: ContrastSet;
  fit: FitResponse;
  conversations: StaticConversation[];
}

export interface AppConfig {
  apiBase: string;
}
