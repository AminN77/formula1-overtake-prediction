export function FeatureImpact({
  impacts,
  className = "",
}: {
  impacts: { feature: string; max_abs_delta_probability: number }[];
  className?: string;
}) {
  if (!impacts?.length) return null;
  return (
    <div className={`rounded-xl border border-white/10 bg-f1-surface/40 p-4 ${className}`}>
      <h4 className="text-sm font-bold text-white">Local sensitivity (numeric features)</h4>
      <p className="mt-1 text-xs text-f1-muted">
        Approximate |ΔP| from a small bump — which inputs move the score most in feature space.
      </p>
      <ul className="mt-3 space-y-2">
        {impacts.map((i) => (
          <li key={i.feature} className="flex justify-between text-sm">
            <span className="text-f1-muted">{i.feature}</span>
            <span className="font-mono text-f1-red">{i.max_abs_delta_probability.toFixed(7)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
