export function ProbabilityGauge({ p }: { p: number }) {
  const pct = Math.round(p * 1000) / 10;
  return (
    <div className="relative mx-auto h-40 w-40">
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r="42" fill="none" stroke="#2d2d3d" strokeWidth="10" />
        <circle
          cx="50"
          cy="50"
          r="42"
          fill="none"
          stroke="#e10600"
          strokeWidth="10"
          strokeDasharray={`${(pct / 100) * 264} 264`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-3xl font-bold text-white">{pct}%</div>
        <div className="text-xs text-f1-muted">P(overtake)</div>
      </div>
    </div>
  );
}
