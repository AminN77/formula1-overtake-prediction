import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { BattleForm, useSchemaForm } from "../components/battle/BattleForm";
import { FeatureImpact } from "../components/battle/FeatureImpact";
import { PredictionCard } from "../components/battle/PredictionCard";
import { SensitivityChart } from "../components/battle/SensitivityChart";
import type { FeatureSchemaItem, PredictResponse, SchemaResponse } from "../types";

export function SingleBattle() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const { values, setValues } = useSchemaForm(schema?.features ?? null);
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [sensFeature, setSensFeature] = useState<string>("");
  const [sens, setSens] = useState<{
    baseline_probability: number;
    curve: { value: number; probability: number }[];
    feature: string;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.modelsSchema();
        if (!cancelled) setSchema(s);
      } catch (e) {
        if (!cancelled) setErr(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const onPredict = useCallback(async () => {
    setErr(null);
    try {
      const body = { inputs: values, include_impacts: true, include_row: false };
      const r = await api.predictSingle(body);
      setPred(r);
    } catch (e) {
      setErr(String(e));
    }
  }, [values]);

  const onSensitivity = useCallback(async () => {
    if (!sensFeature) return;
    setErr(null);
    try {
      const r = await api.sensitivity({
        inputs: values,
        feature: sensFeature,
        min: undefined,
        max: undefined,
        steps: 32,
      });
      setSens(r);
    } catch (e) {
      setErr(String(e));
    }
  }, [values, sensFeature]);

  const numericFeatures: FeatureSchemaItem[] =
    schema?.features.filter((f) => f.kind === "number") ?? [];

  if (loading) return <div className="text-f1-muted">Loading schema…</div>;
  if (err && !schema) return <div className="text-red-400">{err}</div>;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Battle prediction</h1>
        <p className="mt-2 max-w-2xl text-f1-muted">
          Adjust inputs — the form is driven by the active model schema. Run prediction, then explore
          one-parameter sensitivity.
        </p>
      </div>

      {schema && (
        <BattleForm features={schema.features} values={values} onChange={setValues} />
      )}

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm">{err}</div>}

      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onPredict}
          className="rounded-lg bg-f1-red px-6 py-3 font-bold text-white shadow-lg transition hover:brightness-110"
        >
          Predict
        </button>
      </div>

      {pred && (
        <div className="grid gap-8 lg:grid-cols-2">
          <PredictionCard
            probability={pred.probability}
            threshold={pred.threshold}
            verdict={pred.verdict}
            label={pred.label}
            modelVersion={pred.model_version}
          />
          {pred.impacts && <FeatureImpact impacts={pred.impacts} />}
        </div>
      )}

      {pred && numericFeatures.length > 0 && (
        <div className="space-y-3 rounded-xl border border-white/10 bg-f1-surface/30 p-4">
          <h3 className="text-lg font-semibold text-white">1D sensitivity</h3>
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm text-f1-muted">
              Feature
              <select
                className="ml-2 rounded border border-white/10 bg-f1-card px-2 py-2 text-white"
                value={sensFeature}
                onChange={(e) => setSensFeature(e.target.value)}
              >
                <option value="">Select…</option>
                {numericFeatures.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.name}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={onSensitivity}
              disabled={!sensFeature}
              className="rounded-lg bg-white/10 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
            >
              Run curve
            </button>
          </div>
          {sens && (
            <SensitivityChart
              feature={sens.feature}
              curve={sens.curve}
              baseline={sens.baseline_probability}
            />
          )}
        </div>
      )}
    </div>
  );
}
