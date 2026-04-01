const base = () => (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${base()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<{ status: string }>("/api/health"),
  modelsCurrent: () => apiFetch<Record<string, unknown>>("/api/models/current"),
  modelsSchema: () => apiFetch<import("../types").SchemaResponse>("/api/models/schema"),
  modelsGlobalImportance: () =>
    apiFetch<import("../types").GlobalImportanceResponse>("/api/models/importance"),
  modelsVersions: () => apiFetch<{ versions: string[] }>("/api/models/versions"),
  modelsSwitch: (version: string) =>
    apiFetch<{ active: string }>("/api/models/switch", {
      method: "POST",
      body: JSON.stringify({ version }),
    }),
  circuits: () => apiFetch<import("../types").CircuitsResponse>("/api/circuits"),
  standings: (year: number) =>
    apiFetch<import("../types").StandingsResponse>(`/api/standings?year=${year}`),
  predictSingle: (body: object) =>
    apiFetch<import("../types").PredictResponse>("/api/predict/single", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deriveRow: (inputs: Record<string, unknown>) =>
    apiFetch<import("../types").DeriveRowResponse>("/api/predict/derive", {
      method: "POST",
      body: JSON.stringify({ inputs }),
    }),
  sensitivity: (body: object) =>
    apiFetch<import("../types").SensitivityResponse>("/api/sensitivity", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  predictBatch: async (file: File, threshold: number, filterPits: boolean) => {
    const fd = new FormData();
    fd.append("file", file);
    const q = new URLSearchParams({
      threshold: String(threshold),
      filter_pits: String(filterPits),
    });
    const res = await fetch(`${base()}/api/predict/batch?${q}`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<import("../types").BatchPredictResponse>;
  },
};
