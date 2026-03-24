import { useEffect, useMemo, useState } from "react";
import type { CircuitMeta, FeatureSchemaItem } from "../../types";
import { InfoTooltip } from "../shared/InfoTooltip";
import { TeamBadge } from "../shared/TeamBadge";
import { TyreIcon } from "../shared/TyreIcon";
import { CircuitSelector } from "./CircuitSelector";

const GROUP_TITLES: Record<string, string> = {
  race: "Race",
  positions: "Grid positions",
  attacker: "Attacker",
  defender: "Defender",
  teams: "Constructors",
  track: "Track",
  weather: "Weather",
  flags: "Flags",
  battle: "Battle",
  situation: "Situation",
  other: "Other",
};

function FieldLabel({
  f,
  showTechnical,
}: {
  f: FeatureSchemaItem;
  showTechnical: boolean;
}) {
  const label = f.label || f.name.replace(/_/g, " ");
  return (
    <span className="flex flex-wrap items-center gap-0.5">
      <span className="text-white">{label}</span>
      <InfoTooltip text={f.description} />
      {showTechnical && <span className="block w-full font-mono text-[10px] text-f1-muted/80">{f.name}</span>}
    </span>
  );
}

function ReadonlyField({
  f,
  displayValue,
  showTechnical,
}: {
  f: FeatureSchemaItem;
  displayValue: string;
  showTechnical: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <FieldLabel f={f} showTechnical={showTechnical} />
        <span className="text-lg" title="Derived / fixed for this UI">
          🔒
        </span>
      </div>
      <div className="mt-1 font-mono text-sm text-f1-red">{displayValue}</div>
      {f.derived_from && f.derived_from.length > 0 && (
        <p className="mt-1 text-[10px] text-f1-muted">From: {f.derived_from.join(", ")}</p>
      )}
    </div>
  );
}

function FieldEditor({
  f,
  value,
  onChange,
  showTechnical,
  forceReadonly,
}: {
  f: FeatureSchemaItem;
  value: unknown;
  onChange: (v: unknown) => void;
  showTechnical: boolean;
  forceReadonly?: boolean;
}) {
  const ro = forceReadonly || f.readonly;

  if (f.kind === "boolean") {
    return (
      <label className={`flex items-center gap-2 text-sm ${ro ? "opacity-60" : ""}`}>
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={ro}
          onChange={(e) => onChange(e.target.checked)}
          className="accent-f1-red"
        />
        <FieldLabel f={f} showTechnical={showTechnical} />
      </label>
    );
  }
  if (f.kind === "category" && f.options?.length) {
    const opts = f.options.map(String);
    if (f.name.includes("team")) {
      return (
        <label className="block text-sm">
          <span className="flex items-center gap-1 text-f1-muted">
            <FieldLabel f={f} showTechnical={showTechnical} />
          </span>
          <select
            className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2 disabled:opacity-50"
            value={String(value ?? "")}
            disabled={ro}
            onChange={(e) => onChange(e.target.value)}
          >
            {opts.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
          {String(value) && <TeamBadge name={String(value)} />}
        </label>
      );
    }
    if (f.name.includes("tyre_compound") || f.name.includes("compound")) {
      return (
        <label className="block text-sm">
          <span className="flex items-center gap-1 text-f1-muted">
            <FieldLabel f={f} showTechnical={showTechnical} />
          </span>
          <select
            className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2 disabled:opacity-50"
            value={String(value ?? "")}
            disabled={ro}
            onChange={(e) => onChange(e.target.value)}
          >
            {opts.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
          <TyreIcon compound={String(value)} />
        </label>
      );
    }
    return (
      <label className="block text-sm">
        <span className="flex items-center gap-1 text-f1-muted">
          <FieldLabel f={f} showTechnical={showTechnical} />
        </span>
        <select
          className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2 disabled:opacity-50"
          value={String(value ?? "")}
          disabled={ro}
          onChange={(e) => onChange(e.target.value)}
        >
          {opts.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const numVal = typeof value === "number" ? value : Number(value);
  return (
    <label className="block text-sm">
      <span className="flex items-center gap-1 text-f1-muted">
        <FieldLabel f={f} showTechnical={showTechnical} />
      </span>
      <input
        type="number"
        step="any"
        className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2 disabled:opacity-50"
        value={Number.isFinite(numVal) ? numVal : 0}
        min={f.min ?? undefined}
        max={f.max ?? undefined}
        disabled={ro}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function BattleForm({
  features,
  values,
  onChange,
  circuits,
  raceName,
  onRaceChange,
  uiYear = 2025,
}: {
  features: FeatureSchemaItem[];
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  circuits: CircuitMeta[] | null;
  raceName: string;
  onRaceChange: (name: string, meta: CircuitMeta) => void;
  uiYear?: number;
}) {
  const [advanced, setAdvanced] = useState(false);

  const byName = useMemo(() => {
    const m: Record<string, FeatureSchemaItem> = {};
    for (const f of features) m[f.name] = f;
    return m;
  }, [features]);

  const raceProgressPct = useMemo(() => {
    const lap = Number(values.lap_number ?? 0);
    const total = Number(values.total_laps ?? 1);
    if (!total || total <= 0) return 0;
    return Math.round((lap / total) * 1000) / 10;
  }, [values.lap_number, values.total_laps]);

  const setField = (name: string, v: unknown) => {
    onChange({ ...values, [name]: v });
  };

  const handleCircuit = (name: string, meta: CircuitMeta) => {
    onRaceChange(name, meta);
    onChange({
      ...values,
      race_name: name,
      round_number: meta.round,
      total_laps: meta.total_laps,
      year: uiYear,
    });
  };

  const groups = useMemo(() => {
    const g: Record<string, FeatureSchemaItem[]> = {};
    for (const f of features) {
      const key = f.group || "other";
      g[key] = g[key] || [];
      g[key].push(f);
    }
    return g;
  }, [features]);

  const basicFeatures = useMemo(() => features.filter((f) => !f.advanced), [features]);

  const formatReadonlyValue = (f: FeatureSchemaItem): string => {
    const v = values[f.name];
    if (f.name === "race_progress" && typeof v === "number") return `${(v * 100).toFixed(1)}%`;
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : "—";
    return String(v ?? "—");
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-f1-surface/40 px-4 py-3">
        <div className="text-sm text-f1-muted">
          <span className="font-semibold text-white">Input mode</span> — Basic keeps common battle inputs; Advanced exposes
          all model features.
        </div>
        <div className="flex rounded-full border border-white/10 p-0.5">
          <button
            type="button"
            onClick={() => setAdvanced(false)}
            className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
              !advanced ? "bg-f1-red text-white" : "text-f1-muted hover:text-white"
            }`}
          >
            Basic
          </button>
          <button
            type="button"
            onClick={() => setAdvanced(true)}
            className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
              advanced ? "bg-f1-red text-white" : "text-f1-muted hover:text-white"
            }`}
          >
            Advanced
          </button>
        </div>
      </div>

      <CircuitSelector circuits={circuits} value={raceName} onSelect={handleCircuit} />

      <div className="flex flex-wrap gap-2">
        <span className="inline-flex items-center gap-1 rounded-full border border-f1-red/40 bg-f1-red/10 px-3 py-1 text-xs font-semibold text-white">
          Season {uiYear}
          <InfoTooltip text="Locked to 2025 in this app." />
        </span>
        {byName.round_number && (
          <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-black/30 px-3 py-1 text-xs text-f1-muted">
            Round <span className="font-mono text-white">{String(values.round_number ?? "—")}</span>
            <InfoTooltip text={byName.round_number.description || ""} />
          </span>
        )}
        {byName.total_laps && (
          <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-black/30 px-3 py-1 text-xs text-f1-muted">
            Laps <span className="font-mono text-white">{String(values.total_laps ?? "—")}</span>
            <InfoTooltip text={byName.total_laps.description || ""} />
          </span>
        )}
        {byName.race_progress && (
          <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-black/30 px-3 py-1 text-xs text-f1-muted">
            Progress <span className="font-mono text-f1-red">{raceProgressPct}%</span>
            <InfoTooltip text="lap ÷ total laps — recomputed on the server from your inputs." />
          </span>
        )}
      </div>

      {!advanced && (
        <div className="space-y-6">
          <div className="rounded-xl border border-f1-red/20 bg-gradient-to-br from-f1-surface/80 to-f1-card/40 p-4">
            <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-f1-red">Quick inputs</h3>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {basicFeatures.map((f) => (
                <FieldEditor
                  key={f.name}
                  f={f}
                  value={values[f.name]}
                  onChange={(v) => setField(f.name, v)}
                  showTechnical={false}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {advanced && (
        <div className="space-y-6">
          {Object.entries(groups).map(([gk, fs]) => (
            <div key={gk} className="rounded-xl border border-white/10 bg-f1-surface/50 p-4">
              <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-f1-red">
                {GROUP_TITLES[gk] || gk}
              </h3>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {fs.map((f) =>
                  f.readonly ? (
                    <ReadonlyField
                      key={f.name}
                      f={f}
                      displayValue={formatReadonlyValue(f)}
                      showTechnical
                    />
                  ) : (
                    <FieldEditor
                      key={f.name}
                      f={f}
                      value={values[f.name]}
                      onChange={(v) => setField(f.name, v)}
                      showTechnical
                    />
                  ),
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function useSchemaForm(features: FeatureSchemaItem[] | null) {
  const [values, setValues] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (!features?.length) return;
    const init: Record<string, unknown> = {};
    for (const f of features) {
      init[f.name] = f.default ?? (f.kind === "boolean" ? false : 0);
    }
    init.year = 2025;
    setValues(init);
  }, [features]);

  return { values, setValues };
}
