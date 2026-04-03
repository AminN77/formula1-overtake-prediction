import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type {
  CircuitMeta,
  GlobalImportanceResponse,
  ModelCurrentResponse,
  SchemaResponse,
} from "../types";

type AppDataContextValue = {
  currentModel: ModelCurrentResponse | null;
  schema: SchemaResponse | null;
  circuits: CircuitMeta[] | null;
  versions: string[];
  globalImportance: GlobalImportanceResponse["importance"];
  loading: boolean;
  switchingModel: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  switchModel: (version: string) => Promise<void>;
};

const AppDataContext = createContext<AppDataContextValue | null>(null);

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [currentModel, setCurrentModel] = useState<ModelCurrentResponse | null>(null);
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [circuits, setCircuits] = useState<CircuitMeta[] | null>(null);
  const [versions, setVersions] = useState<string[]>([]);
  const [globalImportance, setGlobalImportance] = useState<GlobalImportanceResponse["importance"]>([]);
  const [loading, setLoading] = useState(true);
  const [switchingModel, setSwitchingModel] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [current, schemaResp, versionResp, circuitResp, importanceResp] = await Promise.all([
        api.modelsCurrent(),
        api.modelsSchema(),
        api.modelsVersions(),
        api.circuits(),
        api.modelsGlobalImportance(),
      ]);
      setCurrentModel(current);
      setSchema(schemaResp);
      setVersions(versionResp.versions);
      setCircuits(circuitResp.circuits);
      setGlobalImportance(importanceResp.importance);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const switchModel = useCallback(
    async (version: string) => {
      if (version === currentModel?.version) return;
      setSwitchingModel(true);
      setError(null);
      try {
        await api.modelsSwitch(version);
        await refresh();
      } catch (e) {
        setError(String(e));
        throw e;
      } finally {
        setSwitchingModel(false);
      }
    },
    [currentModel?.version, refresh],
  );

  const value = useMemo(
    () => ({
      currentModel,
      schema,
      circuits,
      versions,
      globalImportance,
      loading,
      switchingModel,
      error,
      refresh,
      switchModel,
    }),
    [circuits, currentModel, error, globalImportance, loading, refresh, schema, switchModel, switchingModel, versions],
  );

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}

export function useAppData(): AppDataContextValue {
  const ctx = useContext(AppDataContext);
  if (!ctx) throw new Error("useAppData must be used within AppDataProvider");
  return ctx;
}
