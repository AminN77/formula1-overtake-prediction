import type { CircuitMeta } from "../../types";

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
  onSelect,
}: {
  circuits: CircuitMeta[] | null;
  value: string;
  onSelect: (raceName: string, meta: CircuitMeta) => void;
}) {
  const meta = circuits?.find((c) => c.race_name === value) ?? null;

  return (
    <div className="space-y-3">
      <label className="block text-sm">
        <span className="font-medium text-white">Circuit (2025)</span>
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
  );
}
