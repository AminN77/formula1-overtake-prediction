import { ProbabilityGauge } from "../shared/ProbabilityGauge";

export function PredictionCard({
  probability,
  threshold,
  verdict,
  label,
  modelVersion,
  attackerConstructorRank,
  defenderConstructorRank,
  constructorRankDelta,
}: {
  probability: number;
  threshold: number;
  verdict: string;
  label: string;
  modelVersion: string;
  attackerConstructorRank?: number;
  defenderConstructorRank?: number;
  constructorRankDelta?: number;
}) {
  const over = probability >= threshold;
  const hasConstructorRanks =
    Number.isFinite(attackerConstructorRank) && Number.isFinite(defenderConstructorRank);

  return (
    <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-f1-surface to-f1-card p-6 shadow-xl">
      <div className="text-center text-xs uppercase tracking-widest text-f1-muted">Prediction</div>
      <ProbabilityGauge p={probability} />
      <div className="mt-4 text-center text-lg font-semibold text-white">{label}</div>
      <div className="mt-2 flex justify-center">
        <span
          className={`rounded-full px-4 py-1 text-sm font-bold ${
            over ? "bg-f1-red text-white" : "bg-white/10 text-f1-muted"
          }`}
        >
          {verdict === "overtake" ? "Predicted overtake" : "No overtake (vs threshold)"}
        </span>
      </div>
      <div className="mt-4 text-center text-xs text-f1-muted">
        Model {modelVersion} · threshold {threshold.toFixed(3)}
      </div>
      {hasConstructorRanks && (
        <div className="mt-4 rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="text-center text-[11px] uppercase tracking-widest text-f1-muted">Before-event constructor standings</div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[11px] text-f1-muted">Attacker</div>
              <div className="font-mono text-base text-white">P{attackerConstructorRank}</div>
            </div>
            <div>
              <div className="text-[11px] text-f1-muted">Defender</div>
              <div className="font-mono text-base text-white">P{defenderConstructorRank}</div>
            </div>
            <div>
              <div className="text-[11px] text-f1-muted">Delta</div>
              <div className="font-mono text-base text-white">
                {Number.isFinite(constructorRankDelta) ? String(constructorRankDelta) : "—"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
