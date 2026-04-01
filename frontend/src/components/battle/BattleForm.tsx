import { useEffect, useMemo, useState } from "react";
import type { CircuitMeta, FeatureSchemaItem } from "../../types";
import { InfoTooltip } from "../shared/InfoTooltip";
import { TeamBadge } from "../shared/TeamBadge";
import { TyreIcon } from "../shared/TyreIcon";
import { CircuitSelector } from "./CircuitSelector";

const INPUT_CLASS =
  "mt-1.5 w-full rounded-lg border border-white/15 bg-f1-card px-3 py-2.5 text-sm text-white shadow-inner transition placeholder:text-f1-muted/50 focus:border-f1-red/45 focus:outline-none focus:ring-2 focus:ring-f1-red/20 disabled:cursor-not-allowed disabled:opacity-45";

/** Hint text for numeric fields: example from schema default + allowed range when present. */
function placeholderForNumberField(f: FeatureSchemaItem): string {
  const parts: string[] = [];
  const d = f.default;
  if (typeof d === "number" && Number.isFinite(d)) {
    parts.push(`e.g. ${d}`);
  }
  const { min: mn, max: mx } = f;
  if (mn != null && mx != null) {
    parts.push(`${mn}–${mx}`);
  } else if (mn != null) {
    parts.push(`≥ ${mn}`);
  } else if (mx != null) {
    parts.push(`≤ ${mx}`);
  }
  if (parts.length) return parts.join(" · ");
  return "Enter a number";
}

function emptySelectLabel(f: FeatureSchemaItem): string {
  const n = f.name.toLowerCase();
  if (n.includes("team")) return "Select a constructor…";
  if (n.includes("compound") || n.includes("tyre_compound")) return "Select compound…";
  if (n === "race_name") return "Select a race…";
  if (n.includes("stint_phase")) return "Select phase…";
  return "Select…";
}

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

function readonlyHelpText(f: FeatureSchemaItem, byName: Record<string, FeatureSchemaItem>): string {
  if (f.derived_from?.length) {
    const labels = f.derived_from.map((n) => byName[n]?.label || n.replace(/_/g, " "));
    return `This value updates when you edit: ${labels.join(", ")}. It is recomputed from your inputs (and reflected here in real time).`;
  }
  if (f.description) return f.description;
  return "Derived or fixed for this form.";
}

function ReadonlyField({
  f,
  displayValue,
  showTechnical,
  byName,
}: {
  f: FeatureSchemaItem;
  displayValue: string;
  showTechnical: boolean;
  byName: Record<string, FeatureSchemaItem>;
}) {
  const help = readonlyHelpText(f, byName);
  return (
    <div className="rounded-lg border border-amber-500/20 bg-gradient-to-br from-black/30 to-f1-surface/30 px-3 py-2.5">
      <div className="flex items-start justify-between gap-2">
        <FieldLabel f={f} showTechnical={showTechnical} />
        <InfoTooltip text={help} />
      </div>
      <div className="mt-1.5 rounded-md border border-white/5 bg-black/25 px-2.5 py-2 font-mono text-sm tabular-nums text-f1-red">
        {displayValue}
      </div>
      {f.derived_from && f.derived_from.length > 0 && (
        <p className="mt-2 text-[11px] leading-snug text-f1-muted">
          Driven by:{" "}
          <span className="text-white/90">
            {f.derived_from.map((n) => byName[n]?.label || n).join(" · ")}
          </span>
        </p>
      )}
    </div>
  );
}

function valueToNumberText(v: unknown): string {
  if (v === "" || v === null || v === undefined) return "";
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  return "";
}

/** Plain text — no spinners; commits `number` or `""` on blur; allows free typing in between. */
function NumberFieldEditor({
  f,
  value,
  onChange,
  showTechnical,
  disabled,
}: {
  f: FeatureSchemaItem;
  value: unknown;
  onChange: (v: unknown) => void;
  showTechnical: boolean;
  disabled: boolean;
}) {
  const [text, setText] = useState(() => valueToNumberText(value));

  useEffect(() => {
    setText(valueToNumberText(value));
  }, [f.name, value]);

  return (
    <label className="block text-sm">
      <span className="flex items-center gap-1 text-f1-muted">
        <FieldLabel f={f} showTechnical={showTechnical} />
      </span>
      <input
        type="text"
        inputMode="decimal"
        autoComplete="off"
        spellCheck={false}
        className={INPUT_CLASS}
        disabled={disabled}
        placeholder={placeholderForNumberField(f)}
        title={f.description || undefined}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          const t = text.trim();
          if (t === "" || t === "-" || t === "." || t === "-.") {
            setText("");
            onChange("");
            return;
          }
          const n = parseFloat(t);
          if (Number.isFinite(n)) {
            onChange(n);
            setText(String(n));
          } else {
            setText("");
            onChange("");
          }
        }}
      />
    </label>
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
      <label
        className={`flex cursor-pointer items-start gap-3 rounded-lg border border-white/10 bg-f1-surface/30 px-3 py-2.5 text-sm transition hover:border-white/20 ${ro ? "opacity-60" : ""}`}
      >
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={ro}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-0.5 accent-f1-red"
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
            className={INPUT_CLASS}
            value={String(value ?? "")}
            disabled={ro}
            title={f.description || undefined}
            onChange={(e) => onChange(e.target.value)}
          >
            <option value="">{emptySelectLabel(f)}</option>
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
            className={INPUT_CLASS}
            value={String(value ?? "")}
            disabled={ro}
            title={f.description || undefined}
            onChange={(e) => onChange(e.target.value)}
          >
            <option value="">{emptySelectLabel(f)}</option>
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
          className={INPUT_CLASS}
          value={String(value ?? "")}
          disabled={ro}
          title={f.description || undefined}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">{emptySelectLabel(f)}</option>
          {opts.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  }
  return (
    <NumberFieldEditor
      f={f}
      value={value}
      onChange={onChange}
      showTechnical={showTechnical}
      disabled={ro}
    />
  );
}

export function BattleForm({
  features,
  values,
  onChange,
  circuits,
  raceName,
  onRaceChange,
  advanced,
  onAdvancedChange,
  uiYear = 2025,
}: {
  features: FeatureSchemaItem[];
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  circuits: CircuitMeta[] | null;
  raceName: string;
  onRaceChange: (name: string, meta: CircuitMeta) => void;
  advanced: boolean;
  onAdvancedChange: (advanced: boolean) => void;
  uiYear?: number;
}) {

  const byName = useMemo(() => {
    const m: Record<string, FeatureSchemaItem> = {};
    for (const f of features) m[f.name] = f;
    return m;
  }, [features]);

  const raceProgressPct = useMemo(() => {
    const lapRaw = values.lap_number;
    const lap = typeof lapRaw === "number" && Number.isFinite(lapRaw) ? lapRaw : Number(lapRaw);
    const total = Number(values.total_laps ?? 1);
    if (!Number.isFinite(lap) || !total || total <= 0) return null;
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
    if (typeof v === "boolean") return v ? "true" : "false";
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
            onClick={() => onAdvancedChange(false)}
            className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
              !advanced ? "bg-f1-red text-white" : "text-f1-muted hover:text-white"
            }`}
          >
            Basic
          </button>
          <button
            type="button"
            onClick={() => onAdvancedChange(true)}
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
            Progress{" "}
            <span className="font-mono text-f1-red">{raceProgressPct != null ? `${raceProgressPct}%` : "—"}</span>
            <InfoTooltip text="lap ÷ total laps — recomputed on the server from your inputs." />
          </span>
        )}
      </div>

      {!advanced && (
        <div className="space-y-6">
          <div className="rounded-xl border border-f1-red/20 bg-gradient-to-br from-f1-surface/80 to-f1-card/40 p-5 shadow-lg shadow-black/20">
            <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-f1-red">Quick inputs</h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
                      byName={byName}
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

