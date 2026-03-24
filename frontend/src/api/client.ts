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
  circuits: () => apiFetch<import("../types").CircuitsResponse>("/api/circuits"),
  predictSingle: (body: object) =>
    apiFetch<import("../types").PredictResponse>("/api/predict/single", {
      method: "POST",
      body: JSON.stringify(body),
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
    return res.json() as Promise<Record<string, unknown>>;
  },
};
