import { ProbabilityGauge } from "../shared/ProbabilityGauge";

export function PredictionCard({
  probability,
  threshold,
  verdict,
  label,
  modelVersion,
}: {
  probability: number;
  threshold: number;
  verdict: string;
  label: string;
  modelVersion: string;
}) {
  const over = probability >= threshold;
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
    </div>
  );
}
