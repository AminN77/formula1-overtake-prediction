import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { PredictResponse, SchemaResponse } from "../types";
import { buildInitialValuesFromFeatures } from "../utils/battleFormValues";

export type SensitivityState = {
  baseline_probability: number;
  curve: { value: number; probability: number }[];
  feature: string;
} | null;

type BattlePageContextValue = {
  values: Record<string, unknown>;
  setValues: Dispatch<SetStateAction<Record<string, unknown>>>;
  raceName: string;
  setRaceName: Dispatch<SetStateAction<string>>;
  pred: PredictResponse | null;
  setPred: Dispatch<SetStateAction<PredictResponse | null>>;
  advancedMode: boolean;
  setAdvancedMode: Dispatch<SetStateAction<boolean>>;
  sens: SensitivityState;
  setSens: Dispatch<SetStateAction<SensitivityState>>;
  sensFeature: string;
  setSensFeature: Dispatch<SetStateAction<string>>;
  analysisTab: "local" | "ranked";
  setAnalysisTab: Dispatch<SetStateAction<"local" | "ranked">>;
  syncInitialValuesFromSchema: (schema: SchemaResponse) => void;
};

const BattlePageContext = createContext<BattlePageContextValue | null>(null);

export function BattlePageProvider({ children }: { children: ReactNode }) {
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [raceName, setRaceName] = useState("");
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [advancedMode, setAdvancedMode] = useState(false);
  const [sens, setSens] = useState<SensitivityState>(null);
  const [sensFeature, setSensFeature] = useState("");
  const [analysisTab, setAnalysisTab] = useState<"local" | "ranked">("local");
  const lastModelVersionRef = useRef<string | undefined>(undefined);

  const syncInitialValuesFromSchema = useCallback((schema: SchemaResponse) => {
    const mv = schema.model_version;
    if (lastModelVersionRef.current === mv) return;
    lastModelVersionRef.current = mv;
    setValues(buildInitialValuesFromFeatures(schema.features));
    setPred(null);
    setSens(null);
    setSensFeature("");
  }, []);

  const value = useMemo(
    () => ({
      values,
      setValues,
      raceName,
      setRaceName,
      pred,
      setPred,
      advancedMode,
      setAdvancedMode,
      sens,
      setSens,
      sensFeature,
      setSensFeature,
      analysisTab,
      setAnalysisTab,
      syncInitialValuesFromSchema,
    }),
    [
      values,
      raceName,
      pred,
      advancedMode,
      sens,
      sensFeature,
      analysisTab,
      syncInitialValuesFromSchema,
    ],
  );

  return <BattlePageContext.Provider value={value}>{children}</BattlePageContext.Provider>;
}

export function useBattlePageState(): BattlePageContextValue {
  const ctx = useContext(BattlePageContext);
  if (!ctx) throw new Error("useBattlePageState must be used within BattlePageProvider");
  return ctx;
}
