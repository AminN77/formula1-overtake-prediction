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
};

export type GlobalImportanceResponse = {
  model_version: string;
  importance: { feature: string; importance: number }[];
};

export type BatchPredictResponse = {
  summary: Record<string, unknown>;
  evaluation: BatchEvaluation | null;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  csv_base64: string;
};
