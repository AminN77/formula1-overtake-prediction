export type FeatureSchemaItem = {
  name: string;
  kind: "number" | "boolean" | "string" | "category";
  default: unknown;
  min: number | null;
  max: number | null;
  options: unknown[] | null;
  group: string | null;
};

export type SchemaResponse = {
  model_version: string;
  features: FeatureSchemaItem[];
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
