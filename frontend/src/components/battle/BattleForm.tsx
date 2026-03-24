import { useEffect, useMemo, useState } from "react";
import type { FeatureSchemaItem } from "../../types";
import { TeamBadge } from "../shared/TeamBadge";
import { TyreIcon } from "../shared/TyreIcon";

function FieldEditor({
  f,
  value,
  onChange,
}: {
  f: FeatureSchemaItem;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (f.kind === "boolean") {
    return (
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="accent-f1-red"
        />
        {f.name}
      </label>
    );
  }
  if (f.kind === "category" && f.options?.length) {
    const opts = f.options.map(String);
    if (f.name.includes("team")) {
      return (
        <label className="block text-sm">
          <span className="text-f1-muted">{f.name}</span>
          <select
            className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2"
            value={String(value ?? "")}
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
          <span className="text-f1-muted">{f.name}</span>
          <select
            className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2"
            value={String(value ?? "")}
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
        <span className="text-f1-muted">{f.name}</span>
        <select
          className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2"
          value={String(value ?? "")}
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
      <span className="text-f1-muted">{f.name}</span>
      <input
        type="number"
        step="any"
        className="mt-1 w-full rounded border border-white/10 bg-f1-card px-2 py-2"
        value={Number.isFinite(numVal) ? numVal : 0}
        min={f.min ?? undefined}
        max={f.max ?? undefined}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function BattleForm({
  features,
  values,
  onChange,
}: {
  features: FeatureSchemaItem[];
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const groups = useMemo(() => {
    const g: Record<string, FeatureSchemaItem[]> = {};
    for (const f of features) {
      const key = f.group || "other";
      g[key] = g[key] || [];
      g[key].push(f);
    }
    return g;
  }, [features]);

  const setField = (name: string, v: unknown) => {
    onChange({ ...values, [name]: v });
  };

  return (
    <div className="space-y-6">
      {Object.entries(groups).map(([gk, fs]) => (
        <div key={gk} className="rounded-xl border border-white/10 bg-f1-surface/50 p-4">
          <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-f1-red">{gk}</h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {fs.map((f) => (
              <FieldEditor key={f.name} f={f} value={values[f.name]} onChange={(v) => setField(f.name, v)} />
            ))}
          </div>
        </div>
      ))}
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
    setValues(init);
  }, [features]);

  return { values, setValues };
}
