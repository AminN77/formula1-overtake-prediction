import { useEffect, useMemo, useState } from "react";
import { ProbabilityGauge } from "../shared/ProbabilityGauge";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "battle", label: "Race / Battle" },
  { id: "advanced", label: "Advanced" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "—";
  return String(v);
}

export function BatchRowDetailModal({
  open,
  row,
  threshold,
  labelColumn,
  onClose,
}: {
  open: boolean;
  row: Record<string, unknown> | null;
  threshold: number;
  labelColumn: string | null;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<TabId>("overview");

  useEffect(() => {
    if (open) setTab("overview");
  }, [open, row]);

  const overviewFields = useMemo(
    () => [
      ["Outcome", row?.eval_outcome],
      ["Predicted", row?.overtake_predicted],
      ["Actual", labelColumn ? row?.[labelColumn] : null],
      ["Attacker", row?.attacker],
      ["Defender", row?.defender],
      ["Attacker constructor rank (before event)", row?.attacker_constructor_rank],
      ["Defender constructor rank (before event)", row?.defender_constructor_rank],
      ["Constructor rank delta", row?.constructor_rank_delta],
      ["Race", row?.race_name],
      ["Track", row?.track],
      ["Lap", row?.lap_number],
    ],
    [labelColumn, row],
  );

  const battleKeys = [
    "year",
    "race_name",
    "round_number",
    "event_date",
    "lap_number",
    "total_laps",
    "race_progress",
    "attacker",
    "defender",
    "attacker_position",
    "defender_position",
    "attacker_constructor_rank",
    "defender_constructor_rank",
    "constructor_rank_delta",
    "gap_ahead",
    "pace_delta",
    "track",
    "race_phase",
    "laps_remaining",
  ];

  const battleEntries = useMemo(
    () => battleKeys.filter((key) => row && key in row).map((key) => [key, row?.[key]] as const),
    [row],
  );

  const advancedEntries = useMemo(
    () =>
      row
        ? Object.entries(row).filter(([key]) => !battleKeys.includes(key) && key !== "overtake_probability")
        : [],
    [row],
  );

  if (!open || !row) return null;

  const probability = typeof row.overtake_probability === "number" ? row.overtake_probability : Number(row.overtake_probability ?? 0);
  const predictedPositive = Number(row.overtake_predicted ?? 0) === 1;
  const actualPositive = labelColumn ? Number(row[labelColumn] ?? 0) === 1 : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="max-h-[90vh] w-full max-w-5xl overflow-hidden rounded-2xl border border-white/10 bg-f1-surface shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div>
            <h3 className="text-lg font-bold text-white">Batch row detail</h3>
            <p className="text-sm text-f1-muted">
              {formatCell(row.attacker)} vs {formatCell(row.defender)} · {formatCell(row.race_name)} · lap {formatCell(row.lap_number)}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-white/10 px-3 py-2 text-sm text-f1-muted hover:bg-white/5 hover:text-white"
          >
            Close
          </button>
        </div>

        <div className="flex flex-wrap gap-2 border-b border-white/10 px-4 py-3">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition ${
                tab === item.id ? "bg-f1-red text-white" : "bg-white/5 text-f1-muted hover:bg-white/10 hover:text-white"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="max-h-[calc(90vh-8rem)] overflow-y-auto p-5">
          {tab === "overview" && (
            <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
              <div className="rounded-2xl border border-white/10 bg-f1-card/60 p-5">
                <div className="text-center text-xs uppercase tracking-widest text-f1-muted">Prediction</div>
                <ProbabilityGauge p={Number.isFinite(probability) ? probability : 0} />
                <div className="mt-3 text-center text-sm text-f1-muted">Threshold {threshold.toFixed(3)}</div>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  <span className={`rounded-full px-3 py-1 text-xs font-bold ${predictedPositive ? "bg-f1-red text-white" : "bg-white/10 text-f1-muted"}`}>
                    {predictedPositive ? "Predicted true" : "Predicted false"}
                  </span>
                  {actualPositive !== null && (
                    <span className={`rounded-full px-3 py-1 text-xs font-bold ${actualPositive ? "bg-emerald-500/20 text-emerald-100" : "bg-slate-500/20 text-slate-100"}`}>
                      {actualPositive ? "Actual positive" : "Actual negative"}
                    </span>
                  )}
                  {row.eval_outcome != null && (
                    <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-bold text-white">{formatCell(row.eval_outcome)}</span>
                  )}
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {overviewFields.map(([label, value]) => (
                  <div key={label} className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
                    <div className="text-[11px] uppercase tracking-wide text-f1-muted">{label}</div>
                    <div className="mt-1 font-mono text-sm text-white">{formatCell(value)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === "battle" && (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {battleEntries.map(([label, value]) => (
                <div key={label} className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
                  <div className="text-[11px] uppercase tracking-wide text-f1-muted">{label}</div>
                  <div className="mt-1 font-mono text-sm text-white">{formatCell(value)}</div>
                </div>
              ))}
            </div>
          )}

          {tab === "advanced" && (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {advancedEntries.map(([label, value]) => (
                <div key={label} className="rounded-xl border border-white/10 bg-f1-card/40 p-4">
                  <div className="text-[11px] uppercase tracking-wide text-f1-muted">{label}</div>
                  <div className="mt-1 break-words font-mono text-sm text-white">{formatCell(value)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
