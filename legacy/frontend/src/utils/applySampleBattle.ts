import type { Dispatch, SetStateAction } from "react";
import type { FeatureSchemaItem } from "../types";
import { SAMPLE_BATTLE_INPUTS } from "../data/sampleBattleInputs";

export function applySampleBattleInputs(
  features: FeatureSchemaItem[],
  setValues: Dispatch<SetStateAction<Record<string, unknown>>>,
  setRaceName: (name: string) => void,
): void {
  const names = new Set(features.map((f) => f.name));
  setValues((prev) => {
    const next = { ...prev };
    for (const [k, v] of Object.entries(SAMPLE_BATTLE_INPUTS)) {
      if (names.has(k)) next[k] = v;
    }
    next.year = 2025;
    return next;
  });
  const rn = SAMPLE_BATTLE_INPUTS.race_name;
  if (typeof rn === "string" && names.has("race_name")) setRaceName(rn);
}
