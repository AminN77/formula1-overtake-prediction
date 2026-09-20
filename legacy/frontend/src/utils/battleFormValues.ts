import type { FeatureSchemaItem } from "../types";

/** Default empty form state for a feature schema (used on first load / model switch). */
export function buildInitialValuesFromFeatures(features: FeatureSchemaItem[]): Record<string, unknown> {
  const init: Record<string, unknown> = {};
  for (const f of features) {
    if (f.readonly) {
      init[f.name] = f.default ?? (f.kind === "boolean" ? false : 0);
    } else if (f.kind === "number") {
      init[f.name] = "";
    } else if (f.kind === "boolean") {
      init[f.name] = false;
    } else if (f.kind === "category") {
      init[f.name] = "";
    } else {
      init[f.name] = f.default ?? "";
    }
  }
  init.year = 2025;
  return init;
}
