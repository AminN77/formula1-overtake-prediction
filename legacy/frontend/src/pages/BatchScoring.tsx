import { useEffect, useMemo } from "react";
import { api } from "../api/client";
import { BatchRowDetailModal } from "../components/batch/BatchRowDetailModal";
import { useAppData } from "../context/AppDataContext";
import { useBatchPageState } from "../context/BatchPageContext";
import type { BatchEvaluation, BatchQueryRequest } from "../types";

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

function parseNumericFilter(value: string): number | null {
  if (!value.trim()) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function inferLabelColumn(summary: Record<string, unknown> | undefined, columns: string[]): string | null {
  const fromSummary = typeof summary?.label_column === "string" ? String(summary.label_column) : null;
  if (fromSummary && columns.includes(fromSummary)) return fromSummary;
  if (columns.includes("overtake")) return "overtake";
  if (columns.includes("label")) return "label";
  return null;
}

export function BatchScoring() {
  const { currentModel, error: appDataError } = useAppData();
  const {
    file,
    setFile,
    threshold,
    setThreshold,
    filterPits,
    setFilterPits,
    loading,
    setLoading,
    err,
    setErr,
    result,
    setResult,
    filterOptions,
    setFilterOptions,
    page,
    setPage,
    pageSize,
    setPageSize,
    filters,
    setFilters,
    selectedRow,
    setSelectedRow,
    modalOpen,
    setModalOpen,
    resetViewerState,
  } = useBatchPageState();

  useEffect(() => {
    if (typeof currentModel?.threshold === "number" && threshold === 0.5 && !file && !result) {
      setThreshold(currentModel.threshold);
    }
  }, [currentModel?.threshold, file, result, setThreshold, threshold]);

  const runQuery = async (request: BatchQueryRequest) => {
    setLoading(true);
    setErr(null);
    try {
      const response = await api.queryBatch(request);
      setResult(response);
      setFilterOptions(response.filter_options);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!result?.result_id) return;
    void runQuery({
      result_id: result.result_id,
      page,
      page_size: pageSize,
      outcome: filters.outcome,
      prediction: filters.prediction,
      attacker: filters.attacker,
      defender: filters.defender,
      race_name: filters.race_name,
      track: filters.track,
      search: filters.search,
      lap_min: parseNumericFilter(filters.lap_min),
      lap_max: parseNumericFilter(filters.lap_max),
      probability_min: parseNumericFilter(filters.probability_min),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    filters.attacker,
    filters.defender,
    filters.lap_max,
    filters.lap_min,
    filters.outcome,
    filters.prediction,
    filters.probability_min,
    filters.race_name,
    filters.search,
    filters.track,
    page,
    pageSize,
    result?.result_id,
  ]);

  const onRun = async () => {
    if (!file) return;
    setLoading(true);
    setErr(null);
    resetViewerState();
    try {
      const response = await api.predictBatch(file, threshold, filterPits, pageSize);
      setResult(response);
      setFilterOptions(response.filter_options);
    } catch (e) {
      setErr(
        String(e).includes("400") || String(e).includes("failed")
          ? `Batch failed — check that the CSV columns match the active model (${currentModel?.version || "current"}) and required fields are present. ${String(e)}`
          : String(e),
      );
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = async () => {
    if (!result?.result_id) return;
    try {
      const blob = await api.downloadBatchCsv(result.result_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(String(e));
    }
  };

  const rows = result?.rows ?? [];
  const evaluation = result?.evaluation as BatchEvaluation | null | undefined;
  const summary = result?.summary;
  const columns = useMemo(() => {
    if (result?.columns?.length) return result.columns;
    if (rows[0]) return Object.keys(rows[0]);
    return [];
  }, [result?.columns, rows]);
  const labelColumn = useMemo(() => inferLabelColumn(summary, columns), [summary, columns]);
  const activeBatchModelVersion = String(summary?.model_version ?? currentModel?.version ?? "");
  const isV6BatchModel = activeBatchModelVersion.startsWith("v6");
  const displayCols = useMemo(() => {
    const preferred = [
      "eval_outcome",
      ...(labelColumn ? [labelColumn] : []),
      "overtake_predicted",
      "overtake_probability",
      "attacker_constructor_rank",
      "defender_constructor_rank",
      "constructor_rank_delta",
    ];
    const rest = columns.filter((c) => !preferred.includes(c) && !(labelColumn && labelColumn !== "overtake" && c === "overtake"));
    return [...preferred.filter((c, idx) => columns.includes(c) && preferred.indexOf(c) === idx), ...rest];
  }, [columns, labelColumn]);

  const setFilterValue = (key: keyof typeof filters, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setPage(1);
    setSelectedRow(null);
  };

  const totalFiltered = result?.filtered_row_count ?? rows.length;
  const pageCount = result?.page_count ?? 1;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">Batch scoring</h1>
          <p className="mt-2 max-w-3xl text-f1-muted">
            Upload a CSV aligned with the <strong>active</strong> model feature schema. Manage model switching from the
            Models tab; this page keeps your last batch state while you navigate.
          </p>
        </div>
        <div className="rounded-lg border border-white/10 bg-f1-card/70 px-4 py-3 text-right">
          <div className="text-xs uppercase tracking-wider text-f1-muted">Active model</div>
          <div className="font-mono text-sm text-white">{currentModel?.version ?? "—"}</div>
        </div>
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
          {file && <div className="mt-1 text-xs text-f1-muted">Current file: {file.name}</div>}
        </label>
        <label className="text-sm">
          Threshold
          <input
            type="number"
            step="0.01"
            className="ml-2 w-24 rounded-lg border border-white/10 bg-f1-card px-2 py-2"
            value={threshold}
            onChange={(e) => setThreshold(Number.parseFloat(e.target.value) || 0)}
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

      {(err || appDataError) && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">{err || appDataError}</div>
      )}

      {result && summary && evaluation && (
        <div className="grid gap-4 xl:grid-cols-[1fr_1.1fr]">
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
              <dd className="font-mono text-white">{String(summary.model_version ?? currentModel?.version ?? "—")}</dd>
              <dt className="text-f1-muted">Predicted positive rate</dt>
              <dd className="font-mono text-white">
                {summary.predicted_positive_rate !== undefined ? `${(Number(summary.predicted_positive_rate) * 100).toFixed(2)}%` : "—"}
              </dd>
              <dt className="text-f1-muted">Viewer rows matched</dt>
              <dd className="font-mono text-white">{totalFiltered}</dd>
              {summary.actual_positive_rate !== undefined && (
                <>
                  <dt className="text-f1-muted">Actual positive rate{labelColumn ? ` (${labelColumn})` : ""}</dt>
                  <dd className="font-mono text-white">{(Number(summary.actual_positive_rate) * 100).toFixed(2)}%</dd>
                </>
              )}
            </dl>
            <button
              type="button"
              disabled={!result.result_id}
              onClick={() => void downloadCsv()}
              className="mt-4 text-sm font-semibold text-f1-red underline disabled:cursor-not-allowed disabled:opacity-40"
            >
              Download full CSV
            </button>
          </div>

          {evaluation.has_labels ? (
            <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
              <h2 className="text-sm font-bold uppercase tracking-wider text-f1-red">
                {isV6BatchModel ? "Evaluation (v6 special confusion view)" : "Evaluation (next-lap confusion matrix)"}
              </h2>
              <p className="mt-1 text-sm text-f1-muted">
                {isV6BatchModel
                  ? "v6 shows the normal binary confusion matrix plus the horizon-specific breakdown for its broader scenario labels."
                  : "v1-v5 are next-lap overtake models, so batch mode shows the standard binary confusion matrix only."}
              </p>
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  ["TP", evaluation.tp ?? 0, "bg-emerald-500/10", "bg-emerald-500/25 ring-1 ring-emerald-300/40"],
                  ["FP", evaluation.fp ?? 0, "bg-amber-500/10", "bg-amber-500/25 ring-1 ring-amber-300/40"],
                  ["TN", evaluation.tn ?? 0, "bg-slate-500/10", "bg-slate-500/25 ring-1 ring-slate-300/40"],
                  ["FN", evaluation.fn ?? 0, "bg-rose-500/10", "bg-rose-500/25 ring-1 ring-rose-300/40"],
                ].map(([name, value, idleCls, activeCls]) => (
                  <button
                    key={String(name)}
                    type="button"
                    onClick={() => setFilterValue("outcome", String(name))}
                    className={`rounded-lg px-2 py-2 text-center transition ${filters.outcome === name ? activeCls : idleCls}`}
                  >
                    <div className="text-xs text-f1-muted">{name}</div>
                    <div className="font-mono text-lg text-white">{value}</div>
                  </button>
                ))}
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
                <div className="mt-4 overflow-x-auto">
                  <div className="text-xs font-semibold text-f1-muted">
                    {isV6BatchModel ? "Base binary confusion matrix" : "Confusion matrix"}
                  </div>
                  <table className="mt-2 w-full border-collapse text-sm">
                    <thead>
                      <tr>
                        <th className="border border-white/10 p-2 text-f1-muted" />
                        {(evaluation.confusion_labels?.cols ?? ["pred 0", "pred 1"]).map((col) => (
                          <th key={col} className="border border-white/10 p-2 text-center text-f1-muted">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.confusion_matrix.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          <td className="border border-white/10 bg-white/5 p-2 text-xs text-f1-muted">
                            {evaluation.confusion_labels?.rows[rowIndex] ?? `row ${rowIndex}`}
                          </td>
                          {row.map((cell, cellIndex) => (
                            <td key={cellIndex} className="border border-white/10 p-2 text-center font-mono text-white">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {isV6BatchModel && evaluation.horizon_breakdown && evaluation.horizon_breakdown.length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <div className="text-xs font-semibold text-f1-muted">Special v6 horizon confusion view</div>
                  <table className="mt-2 w-full border-collapse text-sm">
                    <thead>
                      <tr>
                        <th className="border border-white/10 p-2 text-left text-f1-muted">Horizon</th>
                        <th className="border border-white/10 p-2 text-right text-f1-muted">Labeled positive</th>
                        <th className="border border-white/10 p-2 text-right text-f1-muted">Predicted true</th>
                        <th className="border border-white/10 p-2 text-right text-f1-muted">Hit rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.horizon_breakdown.map((item) => (
                        <tr key={item.column}>
                          <td className="border border-white/10 p-2 text-white">{item.label}</td>
                          <td className="border border-white/10 p-2 text-right font-mono text-white">{item.positive_rows}</td>
                          <td className="border border-white/10 p-2 text-right font-mono text-white">{item.predicted_true}</td>
                          <td className="border border-white/10 p-2 text-right font-mono text-white">
                            {item.predicted_true_rate != null ? `${(item.predicted_true_rate * 100).toFixed(1)}%` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-white/15 bg-f1-surface/30 p-4 text-sm text-f1-muted">
              No recognized target label column was found for the active model. Confusion matrix and classification metrics are skipped.
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-f1-red">CSV viewer filters</h2>
                <p className="mt-1 text-sm text-f1-muted">Filter the full scored result set and page through matching rows.</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  resetViewerState();
                  setFilters((prev) => ({ ...prev }));
                }}
                className="rounded-lg border border-white/10 px-3 py-2 text-sm text-f1-muted hover:bg-white/5 hover:text-white"
              >
                Reset viewer
              </button>
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
              <label className="text-sm">
                <span className="text-f1-muted">Outcome</span>
                <select
                  value={filters.outcome}
                  onChange={(e) => setFilterValue("outcome", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", "TP", "FP", "TN", "FN"].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Prediction</span>
                <select
                  value={filters.prediction}
                  onChange={(e) => setFilterValue("prediction", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", "Predicted positive", "Predicted negative"].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Attacker</span>
                <select
                  value={filters.attacker}
                  onChange={(e) => setFilterValue("attacker", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", ...(filterOptions.attacker ?? [])].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Defender</span>
                <select
                  value={filters.defender}
                  onChange={(e) => setFilterValue("defender", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", ...(filterOptions.defender ?? [])].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Track</span>
                <select
                  value={filters.track}
                  onChange={(e) => setFilterValue("track", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", ...(filterOptions.track ?? [])].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Race</span>
                <select
                  value={filters.race_name}
                  onChange={(e) => setFilterValue("race_name", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                >
                  {["ALL", ...(filterOptions.race_name ?? [])].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Search</span>
                <input
                  type="text"
                  value={filters.search}
                  onChange={(e) => setFilterValue("search", e.target.value)}
                  placeholder="PIA, HAM, Melbourne..."
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                />
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Lap from</span>
                <input
                  type="number"
                  value={filters.lap_min}
                  onChange={(e) => setFilterValue("lap_min", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                />
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Lap to</span>
                <input
                  type="number"
                  value={filters.lap_max}
                  onChange={(e) => setFilterValue("lap_max", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                />
              </label>
              <label className="text-sm">
                <span className="text-f1-muted">Min probability</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={filters.probability_min}
                  onChange={(e) => setFilterValue("probability_min", e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-f1-surface px-3 py-2"
                />
              </label>
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <div className="text-sm text-f1-muted">
                Showing page {result.page} of {pageCount} · {rows.length} rows on this page · {totalFiltered} matching rows
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <label className="text-sm text-f1-muted">
                  Rows per page
                  <select
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setPage(1);
                    }}
                    className="ml-2 rounded-lg border border-white/10 bg-f1-surface px-2 py-2 text-white"
                  >
                    {[10, 25, 50, 100].map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  disabled={page <= 1 || loading}
                  onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                  className="rounded-lg border border-white/10 px-3 py-2 text-sm text-white disabled:opacity-40"
                >
                  Previous
                </button>
                <button
                  type="button"
                  disabled={page >= pageCount || loading}
                  onClick={() => setPage((prev) => Math.min(pageCount, prev + 1))}
                  className="rounded-lg border border-white/10 px-3 py-2 text-sm text-white disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>

            <div className="overflow-auto rounded-xl border border-white/10">
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
                  {rows.map((row, idx) => {
                    const outcome = row.eval_outcome != null ? String(row.eval_outcome) : "";
                    const cls = OUTCOME_STYLES[outcome] ?? "";
                    const absoluteIndex = (page - 1) * pageSize + idx;
                    return (
                      <tr
                        key={`${absoluteIndex}-${String(row.attacker ?? "")}-${String(row.defender ?? "")}`}
                        className={`cursor-pointer border-b border-white/5 transition ${cls}`}
                        onClick={() => {
                          setSelectedRow(row);
                          setModalOpen(true);
                        }}
                      >
                        <td className="whitespace-nowrap px-2 py-1.5 font-mono text-f1-muted">{absoluteIndex}</td>
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
        </div>
      )}

      <BatchRowDetailModal
        open={modalOpen}
        row={selectedRow}
        threshold={Number(summary?.threshold ?? threshold)}
        labelColumn={labelColumn}
        onClose={() => {
          setModalOpen(false);
          setSelectedRow(null);
        }}
      />
    </div>
  );
}
