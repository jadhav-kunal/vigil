export interface Step {
  session_id: string;
  step_index: number;
  model_requested: string;
  model_used: string;
  tool_name: string | null;
  tool_args: Record<string, unknown> | null;
  assistant_text: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  tokens_before_compression: number | null;
  tokens_after_compression: number | null;
  timestamp: string;
  caused_state_mutation: boolean;
  // Watchdog metrics (spec 4.2):
  sim_score: number | null;
  tool_entropy: number | null;
  state_penalty: number | null;
  final_score: number | null;
  watchdog_breach: boolean;
  watchdog_streak: number;
  watchdog_tripped: boolean;
  breaker_override: boolean;
  breaker_state: string | null;
  served_from_cache: boolean;
  // Added by the server on the wire:
  cost_usd: number;
  input_rate_per_1k: number;
  output_rate_per_1k: number;
}

export interface Thresholds {
  theta_sim: number;
  theta_ent: number;
  window: number;
  trip_streak: number;
}

export type BreakerState = "CLOSED" | "HALF_OPEN" | "OPEN";

export interface BreakerInfo {
  state: BreakerState;
  savedEstimate: number;
  tripStepIndex: number | null;
  postMortem: Record<string, unknown> | null;
}

export type ServerMessage =
  | {
      type: "hello";
      price_table: Record<string, [number, number]>;
      thresholds: Thresholds;
    }
  | { type: "snapshot"; count: number }
  | { type: "step"; step: Step }
  | {
      type: "breaker";
      session_id: string;
      state: BreakerState;
      transition: string;
      trip_step_index: number | null;
      saved_estimate_usd: number;
      post_mortem: Record<string, unknown> | null;
    };

export interface SessionAgg {
  id: string;
  steps: Step[];
  cost: number;
  tokensBefore: number;
  tokensAfter: number;
  lastTs: string;
  models: Set<string>;
}

export type ConnState = "connecting" | "open" | "closed";
