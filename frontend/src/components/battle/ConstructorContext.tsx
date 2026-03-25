import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { StandingsResponse } from "../../types";

/** Shows live constructor championship positions from f1api.dev (display only until v6 includes them as features). */
export function ConstructorContext({
  attackerTeam,
  defenderTeam,
  year,
}: {
  attackerTeam: string;
  defenderTeam: string;
  year: number;
}) {
  const [data, setData] = useState<StandingsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setErr(null);
    api
      .standings(year)
      .then((s) => {
        if (!cancelled) setData(s);
      })
      .catch((e) => {
        if (!cancelled) setErr(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [year]);

  const pos = (team: string) => data?.entries.find((e) => e.app_team === team)?.position;
  const pa = attackerTeam ? pos(attackerTeam) : undefined;
  const pd = defenderTeam ? pos(defenderTeam) : undefined;

  if (err) {
    return (
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
        Constructor standings unavailable ({err.slice(0, 120)})
      </div>
    );
  }
  if (!data?.entries.length) return null;

  return (
    <div className="rounded-xl border border-white/10 bg-f1-surface/40 p-3 text-sm">
      <div className="text-xs font-bold uppercase tracking-wider text-f1-red">
        Constructor championship ({data.season}) · {data.source}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {attackerTeam && (
          <span className="rounded-full bg-black/30 px-2 py-1 font-mono text-xs text-white">
            {attackerTeam}: P{pa ?? "?"}
          </span>
        )}
        {defenderTeam && (
          <span className="rounded-full bg-black/30 px-2 py-1 font-mono text-xs text-white">
            {defenderTeam}: P{pd ?? "?"}
          </span>
        )}
        {pa != null && pd != null && (
          <span className="rounded-full border border-f1-red/40 px-2 py-1 text-xs font-semibold text-white">
            Δ rank (att − def): {pa - pd}
          </span>
        )}
      </div>
      <p className="mt-2 text-[11px] text-f1-muted">
        Informational context only — not fed into the current model (v5). Planned for v6 with retraining.
      </p>
    </div>
  );
}
