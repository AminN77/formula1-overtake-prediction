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
