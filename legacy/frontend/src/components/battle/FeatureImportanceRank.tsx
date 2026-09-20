/** Global (model-level) importance from the trained estimator — not local |ΔP|. */
export function FeatureImportanceRank({
  rows,
  className = "",
}: {
  rows: { feature: string; importance: number }[];
  className?: string;
}) {
  if (!rows?.length) return null;
  return (
    <div className={`rounded-xl border border-white/10 bg-f1-surface/40 p-4 ${className}`}>
      <h4 className="text-sm font-bold text-white">Global feature importance</h4>
      <p className="mt-1 text-xs text-f1-muted">
        Trained-model ranking (XGBoost MDI, one-hot columns grouped per original field). Same for any
        battle — shows which inputs carry the most signal at a global scale, not for this scenario only.
      </p>
      <div className="mt-3 max-h-72 overflow-y-auto rounded-lg border border-white/5">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-f1-surface/95 text-xs uppercase tracking-wide text-f1-muted">
            <tr>
              <th className="px-2 py-2">#</th>
              <th className="px-2 py-2">Feature</th>
              <th className="px-2 py-2 text-right">Importance</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={row.feature} className="border-t border-white/5 hover:bg-white/[0.03]">
                <td className="px-2 py-1.5 font-mono text-f1-muted">{idx + 1}</td>
                <td className="px-2 py-1.5 text-f1-muted">{row.feature}</td>
                <td className="px-2 py-1.5 text-right font-mono text-f1-red">
                  {(row.importance * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
