import type { CircuitMeta } from "../../types";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { StandingsResponse } from "../../types";

function flagEmoji(countryCode: string): string {
  const c = countryCode.trim().toUpperCase();
  if (c.length !== 2) return "🏁";
  const A = 0x1f1e6;
  return String.fromCodePoint(
    A + (c.charCodeAt(0) - 65),
    A + (c.charCodeAt(1) - 65),
  );
}

export function CircuitSelector({
  circuits,
  value,
  year,
  attackerTeam,
  defenderTeam,
  onSelect,
}: {
  circuits: CircuitMeta[] | null;
  value: string;
  year: number;
  attackerTeam?: string;
  defenderTeam?: string;
  onSelect: (raceName: string, meta: CircuitMeta) => void;
}) {
  const meta = circuits?.find((c) => c.race_name === value) ?? null;
  const [standings, setStandings] = useState<StandingsResponse | null>(null);
  const [standingsErr, setStandingsErr] = useState<string | null>(null);
  const [loadingStandings, setLoadingStandings] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!meta?.round) return undefined;
    setStandingsErr(null);
    setLoadingStandings(true);
    api
      .standings(year, meta.round, true)
      .then((resp) => {
        if (cancelled) return;
        setStandings(resp);
      })
      .catch((e) => {
        if (cancelled) return;
        setStandings(null);
        setStandingsErr(String(e));
      })
      .finally(() => {
        if (cancelled) return;
        setLoadingStandings(false);
      });
    return () => {
      cancelled = true;
    };
  }, [meta?.round, year]);

  const subtitle = useMemo(() => {
    if (!meta) return "";
    if ((standings?.round_used ?? 0) > 0) {
      return `Before ${meta.race_name} (round ${meta.round}) · standings after round ${standings?.round_used}`;
    }
    return `Before ${meta.race_name} (round ${meta.round})`;
  }, [meta, standings?.round_used]);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
      <div className="space-y-3">
        <label className="block text-sm">
          <span className="font-medium text-white">Circuit ({year})</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-f1-card px-3 py-2.5 text-white"
            value={value}
            onChange={(e) => {
              const name = e.target.value;
              const m = circuits?.find((c) => c.race_name === name);
              if (m) onSelect(name, m);
            }}
          >
            {!circuits?.length && value && (
              <option value={value}>{value.replace(" Grand Prix", " GP")}</option>
            )}
            {!circuits?.length && !value && <option value="">Loading circuits…</option>}
            {circuits?.map((c) => (
              <option key={c.race_name} value={c.race_name}>
                {flagEmoji(c.country)} R{c.round} · {c.race_name.replace(" Grand Prix", " GP")}
              </option>
            ))}
          </select>
        </label>

        {meta && (
          <div className="rounded-xl border border-white/10 bg-f1-surface/60 p-4 text-sm">
            <div className="text-lg font-bold text-white">
              {flagEmoji(meta.country)} {meta.city}
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-f1-muted sm:grid-cols-4">
              <div>
                <dt className="text-xs uppercase tracking-wide">Round</dt>
                <dd className="font-mono text-white">{meta.round}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide">Laps</dt>
                <dd className="font-mono text-white">{meta.total_laps}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide">DRS zones</dt>
                <dd className="font-mono text-white">{meta.drs_zones}</dd>
              </div>
            </dl>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-white/10 bg-f1-surface/60 p-4 text-sm">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-f1-red">Constructor standings</div>
            <div className="mt-1 text-xs text-f1-muted">{subtitle || "Select a circuit to view standings."}</div>
          </div>
          {loadingStandings && <div className="text-xs text-f1-muted">Loading…</div>}
        </div>

        {standingsErr && (
          <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
            Could not load standings. {standingsErr.slice(0, 120)}
          </div>
        )}

        {!standingsErr && !loadingStandings && (standings?.entries?.length ?? 0) === 0 && (
          <div className="mt-3 rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-f1-muted">
            No prior-event standings available for this round yet.
          </div>
        )}

        {(standings?.entries?.length ?? 0) > 0 && (
          <div className="mt-3 overflow-hidden rounded-lg border border-white/10">
            <table className="w-full text-left text-xs">
              <thead className="bg-black/30 text-f1-muted">
                <tr>
                  <th className="px-3 py-2 font-semibold">Pos</th>
                  <th className="px-3 py-2 font-semibold">Team</th>
                  <th className="px-3 py-2 text-right font-semibold">Pts</th>
                  <th className="px-3 py-2 text-right font-semibold">Wins</th>
                </tr>
              </thead>
              <tbody>
                {standings!.entries.map((entry) => (
                  <tr
                    key={`${entry.team_id}-${entry.position}`}
                    className={`border-t border-white/5 ${
                      entry.app_team === attackerTeam
                        ? "bg-f1-red/15"
                        : entry.app_team === defenderTeam
                          ? "bg-sky-500/15"
                          : ""
                    }`}
                  >
                    <td className="px-3 py-2 font-mono text-white">
                      <span
                        className={`inline-flex min-w-[2.6rem] items-center justify-center rounded-full px-2 py-0.5 ${
                          entry.position === 1
                            ? "bg-yellow-400/25 text-yellow-100 ring-1 ring-yellow-200/40"
                            : entry.position === 2
                              ? "bg-slate-200/20 text-slate-100 ring-1 ring-slate-100/30"
                              : entry.position === 3
                                ? "bg-amber-500/20 text-amber-100 ring-1 ring-amber-200/40"
                                : "bg-white/10 text-white"
                        }`}
                      >
                        P{entry.position}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-white">
                      <div className="flex items-center gap-2">
                        <span>{entry.app_team}</span>
                        {entry.app_team === attackerTeam && (
                          <span className="rounded-full border border-f1-red/60 bg-f1-red/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-100">
                            Attacker
                          </span>
                        )}
                        {entry.app_team === defenderTeam && (
                          <span className="rounded-full border border-sky-400/60 bg-sky-400/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-100">
                            Defender
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-white">{entry.points.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono text-white">{entry.wins}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {(attackerTeam || defenderTeam) && (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-f1-muted">
            {attackerTeam && (
              <span className="rounded-full border border-f1-red/40 bg-f1-red/10 px-2 py-1">
                Attacker: <span className="font-semibold text-white">{attackerTeam}</span>
              </span>
            )}
            {defenderTeam && (
              <span className="rounded-full border border-sky-400/40 bg-sky-400/10 px-2 py-1">
                Defender: <span className="font-semibold text-white">{defenderTeam}</span>
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
