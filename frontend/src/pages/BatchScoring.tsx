import { useEffect, useState } from "react";
import { api } from "../api/client";

export function BatchScoring() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [filterPits, setFilterPits] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const m = await api.modelsCurrent();
        if (typeof m.threshold === "number") setThreshold(m.threshold as number);
      } catch {
        /* ignore */
      }
    })();
  }, []);

  const onRun = async () => {
    if (!file) return;
    setLoading(true);
    setErr(null);
    try {
      const r = await api.predictBatch(file, threshold, filterPits);
      setResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = () => {
    if (!result?.csv_base64) return;
    const bin = atob(String(result.csv_base64));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const preview = result?.preview_rows as Record<string, unknown>[] | undefined;
  const summary = result?.summary as Record<string, unknown> | undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Batch scoring</h1>
        <p className="mt-2 text-f1-muted">Upload a battle CSV (e.g. from <code className="text-f1-red">data/v5</code>).</p>
      </div>

      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-white/10 bg-f1-surface/40 p-4">
        <label className="text-sm">
          <span className="text-f1-muted">CSV file</span>
          <input
            type="file"
            accept=".csv"
            className="mt-1 block w-full text-sm"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="text-sm">
          Threshold
          <input
            type="number"
            step="0.01"
            className="ml-2 w-24 rounded border border-white/10 bg-f1-card px-2 py-1"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={filterPits}
            onChange={(e) => setFilterPits(e.target.checked)}
            className="accent-f1-red"
          />
          Filter pit-stop rows
        </label>
        <button
          type="button"
          disabled={!file || loading}
          onClick={onRun}
          className="rounded-lg bg-f1-red px-4 py-2 font-bold text-white disabled:opacity-40"
        >
          {loading ? "Scoring…" : "Score"}
        </button>
      </div>

      {err && <div className="text-red-400">{err}</div>}

      {summary && (
        <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4 text-sm">
          <pre className="whitespace-pre-wrap text-f1-muted">{JSON.stringify(summary, null, 2)}</pre>
          <button type="button" onClick={downloadCsv} className="mt-3 text-f1-red underline">
            Download full CSV
          </button>
        </div>
      )}

      {preview && preview.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-f1-surface text-f1-muted">
              <tr>
                {Object.keys(preview[0]).map((k) => (
                  <th key={k} className="px-3 py-2 font-semibold">
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.slice(0, 20).map((row, i) => (
                <tr key={i} className="border-t border-white/5 hover:bg-white/5">
                  {Object.values(row).map((v, j) => (
                    <td key={j} className="px-3 py-2 font-mono text-xs">
                      {String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
