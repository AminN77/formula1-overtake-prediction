import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { ModelSwitcher } from "../components/model/ModelSwitcher";
import type { BatchEvaluation, BatchPredictResponse } from "../types";

const OUTCOME_STYLES: Record<string, string> = {
  TP: "bg-emerald-500/15 hover:bg-emerald-500/25",
  FP: "bg-amber-500/15 hover:bg-amber-500/25",
  TN: "bg-slate-500/15 hover:bg-slate-500/25",
  FN: "bg-rose-500/15 hover:bg-rose-500/25",
};

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "—";
  return String(v);
}

export function BatchScoring() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [filterPits, setFilterPits] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<BatchPredictResponse | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("");
  const [versions, setVersions] = useState<string[]>([]);
  const [switchingModel, setSwitchingModel] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const refreshModel = useCallback(async () => {
    try {
      const m = await api.modelsCurrent();
      if (typeof m.threshold === "number") setThreshold(m.threshold as number);
      if (typeof m.version === "string") setModelVersion(m.version);
    } catch {
      /* ignore */
    }
    try {
      const v = await api.modelsVersions();
      setVersions(v.versions);
    } catch {
      setVersions([]);
    }
  }, []);

  useEffect(() => {
    void refreshModel();
  }, [refreshModel]);

  const handleModelChange = useCallback(
    async (v: string) => {
      if (v === modelVersion) return;
      setSwitchingModel(true);
      setErr(null);
      setResult(null);
      setSelectedIdx(null);
      try {
        await api.modelsSwitch(v);
        await refreshModel();
      } catch (e) {
        setErr(String(e));
      } finally {
        setSwitchingModel(false);
      }
    },
    [modelVersion, refreshModel],
  );

  const onRun = async () => {
    if (!file) return;
    setLoading(true);
    setErr(null);
    setSelectedIdx(null);
    try {
      const r = await api.predictBatch(file, threshold, filterPits);
      setResult(r);
    } catch (e) {
      setErr(
        String(e).includes("400") || String(e).includes("failed")
          ? `Batch failed — check that the CSV columns match the active model (${modelVersion || "current"}) and required fields are present. ${String(e)}`
          : String(e),
      );
      setResult(null);
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

  const rows = result?.rows ?? [];
  const evaluation = result?.evaluation as BatchEvaluation | null | undefined;
  const summary = result?.summary;
  const columns = useMemo(() => {
    if (result?.columns?.length) return result.columns;
    if (rows[0]) return Object.keys(rows[0]);
    return [];
  }, [result?.columns, rows]);

  const displayCols = useMemo(() => {
    const preferred = ["eval_outcome", "overtake", "overtake_predicted", "overtake_probability"];
    const rest = columns.filter((c) => !preferred.includes(c));
    return [...preferred.filter((c) => columns.includes(c)), ...rest];
  }, [columns]);

  const selectedRow = selectedIdx !== null && rows[selectedIdx] ? rows[selectedIdx] : null;

  return (
    <div className="space-y-6">
      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-3xl font-bold">Batch scoring</h1>
          <ModelSwitcher
            versions={versions}
            active={modelVersion || "—"}
            disabled={switchingModel || loading}
            onChange={handleModelChange}
          />
          {switchingModel && <span className="text-xs text-f1-muted">Switching model…</span>}
        </div>
        <p className="mt-2 text-f1-muted">
          Upload a battle CSV aligned with the <strong>active</strong> model feature schema (e.g.{" "}
          <code className="text-f1-red">data/v5</code> for v5). Switching models clears results until you score again.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-white/10 bg-f1-surface/40 p-4">
        <label className="text-sm">
          <span className="text-f1-muted">CSV file</span>
          <input
            type="file"
            accept=".csv"
            className="mt-1 block w-full max-w-xs text-sm file:mr-2 file:rounded file:border-0 file:bg-f1-red file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-white"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="text-sm">
          Threshold
          <input
            type="number"
            step="0.01"
            className="ml-2 w-24 rounded-lg border border-white/10 bg-f1-card px-2 py-2"
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
          disabled={!file || loading || switchingModel}
          onClick={onRun}
          className="rounded-lg bg-f1-red px-4 py-2 font-bold text-white disabled:opacity-40"
        >
          {loading ? "Scoring…" : "Score"}
        </button>
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">{err}</div>
      )}

      {result && summary && evaluation && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
            <h2 className="text-sm font-bold uppercase tracking-wider text-f1-red">Run summary</h2>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <dt className="text-f1-muted">Rows scored</dt>
              <dd className="font-mono text-white">{String(summary.rows ?? "—")}</dd>
              <dt className="text-f1-muted">Rows in file</dt>
              <dd className="font-mono text-white">{String(summary.rows_input ?? "—")}</dd>
              <dt className="text-f1-muted">Threshold</dt>
              <dd className="font-mono text-white">{String(summary.threshold ?? "—")}</dd>
              <dt className="text-f1-muted">Model</dt>
              <dd className="font-mono text-white">{String(summary.model_version ?? modelVersion)}</dd>
              <dt className="text-f1-muted">Predicted positive rate</dt>
              <dd className="font-mono text-white">
                {summary.predicted_positive_rate !== undefined
                  ? `${(Number(summary.predicted_positive_rate) * 100).toFixed(2)}%`
                  : "—"}
              </dd>
              {summary.actual_positive_rate !== undefined && (
                <>
                  <dt className="text-f1-muted">Actual positive rate</dt>
                  <dd className="font-mono text-white">
                    {(Number(summary.actual_positive_rate) * 100).toFixed(2)}%
                  </dd>
                </>
              )}
              {summary.roc_auc !== undefined && (
                <>
                  <dt className="text-f1-muted">ROC-AUC</dt>
                  <dd className="font-mono text-f1-red">{Number(summary.roc_auc).toFixed(4)}</dd>
                </>
              )}
              {summary.pr_auc !== undefined && (
                <>
                  <dt className="text-f1-muted">PR-AUC</dt>
                  <dd className="font-mono text-f1-red">{Number(summary.pr_auc).toFixed(4)}</dd>
                </>
              )}
            </dl>
            <button type="button" onClick={downloadCsv} className="mt-4 text-sm font-semibold text-f1-red underline">
              Download full CSV
            </button>
          </div>

          {evaluation.has_labels ? (
            <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
              <h2 className="text-sm font-bold uppercase tracking-wider text-f1-red">Evaluation (labeled data)</h2>
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg bg-emerald-500/10 px-2 py-2 text-center">
                  <div className="text-xs text-f1-muted">TP</div>
                  <div className="font-mono text-lg text-white">{evaluation.tp ?? 0}</div>
                </div>
                <div className="rounded-lg bg-amber-500/10 px-2 py-2 text-center">
                  <div className="text-xs text-f1-muted">FP</div>
                  <div className="font-mono text-lg text-white">{evaluation.fp ?? 0}</div>
                </div>
                <div className="rounded-lg bg-slate-500/10 px-2 py-2 text-center">
                  <div className="text-xs text-f1-muted">TN</div>
                  <div className="font-mono text-lg text-white">{evaluation.tn ?? 0}</div>
                </div>
                <div className="rounded-lg bg-rose-500/10 px-2 py-2 text-center">
                  <div className="text-xs text-f1-muted">FN</div>
                  <div className="font-mono text-lg text-white">{evaluation.fn ?? 0}</div>
                </div>
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-2 text-sm">
                <dt className="text-f1-muted">Accuracy</dt>
                <dd className="font-mono">{evaluation.accuracy !== undefined ? evaluation.accuracy.toFixed(4) : "—"}</dd>
                <dt className="text-f1-muted">Precision</dt>
                <dd className="font-mono">{evaluation.precision != null ? evaluation.precision.toFixed(4) : "—"}</dd>
                <dt className="text-f1-muted">Recall</dt>
                <dd className="font-mono">{evaluation.recall != null ? evaluation.recall.toFixed(4) : "—"}</dd>
                <dt className="text-f1-muted">F1</dt>
                <dd className="font-mono">{evaluation.f1 != null ? evaluation.f1.toFixed(4) : "—"}</dd>
              </dl>
              {evaluation.confusion_matrix && evaluation.confusion_matrix.length === 2 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold text-f1-muted">Confusion matrix (actual × predicted)</div>
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full border-collapse text-center text-sm">
                      <thead>
                        <tr>
                          <th className="border border-white/10 p-2 text-f1-muted" />
                          {evaluation.confusion_labels?.cols.map((c) => (
                            <th key={c} className="border border-white/10 p-2 font-mono text-xs text-f1-muted">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {evaluation.confusion_matrix.map((row, i) => (
                          <tr key={i}>
                            <td className="border border-white/10 bg-white/5 p-2 text-xs text-f1-muted">
                              {evaluation.confusion_labels?.rows[i] ?? i}
                            </td>
                            {row.map((cell, j) => (
                              <td key={j} className="border border-white/10 p-2 font-mono text-white">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-white/15 bg-f1-surface/30 p-4 text-sm text-f1-muted">
              No <code className="text-white/80">overtake</code> label column — confusion matrix and classification metrics
              are skipped. Predictions and probabilities are still produced.
            </div>
          )}
        </div>
      )}

      {rows.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-[1fr_minmax(280px,400px)]">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
              <span className="font-semibold text-white">Row colors</span>
              <span className="inline-flex items-center gap-1 rounded bg-emerald-500/20 px-2 py-0.5">TP</span>
              <span className="inline-flex items-center gap-1 rounded bg-amber-500/20 px-2 py-0.5">FP</span>
              <span className="inline-flex items-center gap-1 rounded bg-slate-500/20 px-2 py-0.5">TN</span>
              <span className="inline-flex items-center gap-1 rounded bg-rose-500/20 px-2 py-0.5">FN</span>
              <span className="text-f1-muted">({result?.row_count ?? rows.length} rows)</span>
            </div>
            <div className="max-h-[min(70vh,560px)] overflow-auto rounded-xl border border-white/10">
              <table className="min-w-full text-left text-xs">
                <thead className="sticky top-0 z-10 bg-f1-surface shadow-sm">
                  <tr>
                    <th className="whitespace-nowrap border-b border-white/10 px-2 py-2 text-f1-muted">#</th>
                    {displayCols.map((k) => (
                      <th key={k} className="whitespace-nowrap border-b border-white/10 px-2 py-2 font-semibold text-f1-muted">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const outcome = row.eval_outcome != null ? String(row.eval_outcome) : "";
                    const cls = OUTCOME_STYLES[outcome] ?? "";
                    return (
                      <tr
                        key={i}
                        className={`cursor-pointer border-b border-white/5 transition ${cls} ${
                          selectedIdx === i ? "ring-1 ring-inset ring-f1-red" : ""
                        }`}
                        onClick={() => setSelectedIdx(i)}
                      >
                        <td className="whitespace-nowrap px-2 py-1.5 font-mono text-f1-muted">{i}</td>
                        {displayCols.map((k) => (
                          <td key={k} className="max-w-[12rem] truncate px-2 py-1.5 font-mono text-white/90">
                            {formatCell(row[k])}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="lg:sticky lg:top-4 lg:self-start">
            <div className="rounded-xl border border-white/10 bg-f1-surface/50 p-4 shadow-lg">
              <h3 className="text-sm font-bold uppercase tracking-wider text-f1-red">Row detail</h3>
              {!selectedRow && (
                <p className="mt-3 text-sm text-f1-muted">Click a row in the table to inspect all fields.</p>
              )}
              {selectedRow && (
                <dl className="mt-3 max-h-[min(65vh,520px)] space-y-2 overflow-y-auto pr-1">
                  {Object.entries(selectedRow).map(([k, v]) => (
                    <div key={k} className="rounded-lg border border-white/5 bg-black/20 px-2 py-1.5">
                      <dt className="text-[10px] uppercase tracking-wide text-f1-muted">{k}</dt>
                      <dd className="break-words font-mono text-sm text-white">{formatCell(v)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
