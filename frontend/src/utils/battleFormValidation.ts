import type { FeatureSchemaItem } from "../types";

/** True if the user has not provided a value for an editable field. */
export function isEditableEmpty(f: FeatureSchemaItem, v: unknown): boolean {
  if (f.readonly) return false;
  if (f.kind === "boolean") return false;
  if (f.kind === "number") {
    if (v === "" || v === null || v === undefined) return true;
    if (typeof v === "number") return !Number.isFinite(v);
    return true;
  }
  if (f.kind === "category") {
    return v === "" || v === null || v === undefined;
  }
  return v === "" || v === null || v === undefined;
}

/** Human-readable labels for fields that still need input. */
export function missingEditableFieldLabels(
  features: FeatureSchemaItem[],
  values: Record<string, unknown>,
): string[] {
  const out: string[] = [];
  for (const f of features) {
    if (isEditableEmpty(f, values[f.name])) {
      out.push(f.label || f.name.replace(/_/g, " "));
    }
  }
  return out;
}

export function hasAllEditableFields(features: FeatureSchemaItem[], values: Record<string, unknown>): boolean {
  return missingEditableFieldLabels(features, values).length === 0;
}

/** Fields the user must fill: editable, and in Basic mode only non-advanced fields. */
export function requiredFeaturesForMode(features: FeatureSchemaItem[], advanced: boolean): FeatureSchemaItem[] {
  return features.filter((f) => {
    if (f.readonly) return false;
    if (!advanced && f.advanced) return false;
    return true;
  });
}
