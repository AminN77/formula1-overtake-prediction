export type FeatureSchemaItem = {
  name: string;
  kind: "number" | "boolean" | "string" | "category";
  default: unknown;
  min: number | null;
  max: number | null;
  options: unknown[] | null;
  group: string | null;
  label: string | null;
  description: string | null;
  readonly: boolean;
  advanced: boolean;
  derived_from: string[] | null;
};

export type SchemaResponse = {
  model_version: string;
  features: FeatureSchemaItem[];
  ui_year?: number;
  /** Model artifact feature list; UI may append supplemental inputs for `build_single_row`. */
  trained_feature_names?: string[];
};

export type ModelCurrentResponse = {
  version: string;
  data_version?: string | null;
  threshold?: number;
  train_years?: number[];
  train_rows?: number;
  features_count?: number;
  cv_metrics?: Record<string, unknown> | null;
  meta?: Record<string, unknown>;
};

export type CircuitMeta = {
  race_name: string;
  round: number;
  total_laps: number;
  country: string;
  city: string;
  drs_zones: number;
};

export type CircuitsResponse = {
  season: number;
  circuits: CircuitMeta[];
};

export type StandingsEntry = {
  position: number;
  team_id: string;
  points: number;
  wins: number;
  app_team: string;
  team_name: string | null;
};

export type StandingsResponse = {
  season: number;
  source: string;
  entries: StandingsEntry[];
};

export type PredictResponse = {
  probability: number;
  threshold: number;
  verdict: string;
  label: string;
  model_version: string;
  impacts?: { feature: string; max_abs_delta_probability: number }[];
  row?: Record<string, unknown>;
};

export type SensitivityResponse = {
  baseline_probability: number;
  curve: { value: number; probability: number }[];
  feature: string;
};

export type DeriveRowResponse = {
  row: Record<string, unknown>;
};

export type BatchEvaluation = {
  has_labels: boolean;
  label_column?: string;
  tp?: number;
  fp?: number;
  tn?: number;
  fn?: number;
  accuracy?: number;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  confusion_matrix?: number[][];
  confusion_labels?: { rows: string[]; cols: string[] };
  horizon_breakdown?: {
    column: string;
    label: string;
    positive_rows: number;
    predicted_true: number;
    predicted_true_rate?: number | null;
  }[];
};

export type GlobalImportanceResponse = {
  model_version: string;
  importance: { feature: string; importance: number }[];
};

export type BatchFilterOptions = {
  attacker?: string[];
  defender?: string[];
  race_name?: string[];
  track?: string[];
};

export type BatchPredictResponse = {
  result_id: string;
  summary: Record<string, unknown>;
  evaluation: BatchEvaluation | null;
  columns: string[];
  filter_options: BatchFilterOptions;
  rows: Record<string, unknown>[];
  row_count: number;
  filtered_row_count: number;
  page: number;
  page_size: number;
  page_count: number;
  has_more: boolean;
};

export type BatchQueryRequest = {
  result_id: string;
  page: number;
  page_size: number;
  outcome: string;
  prediction: string;
  attacker: string;
  defender: string;
  race_name: string;
  track: string;
  search: string;
  lap_min: number | null;
  lap_max: number | null;
  probability_min: number | null;
};
