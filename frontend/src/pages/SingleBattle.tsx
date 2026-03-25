import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { BattleForm, useSchemaForm } from "../components/battle/BattleForm";
import { ConstructorContext } from "../components/battle/ConstructorContext";
import { FeatureImpact } from "../components/battle/FeatureImpact";
import { ModelSwitcher } from "../components/model/ModelSwitcher";
import { PredictionCard } from "../components/battle/PredictionCard";
import { SensitivityChart } from "../components/battle/SensitivityChart";
import type { CircuitMeta, FeatureSchemaItem, PredictResponse, SchemaResponse } from "../types";

const UI_YEAR = 2025;

export function SingleBattle() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [versions, setVersions] = useState<string[]>([]);
  const [circuits, setCircuits] = useState<CircuitMeta[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [switchingModel, setSwitchingModel] = useState(false);
  const { values, setValues } = useSchemaForm(schema?.features ?? null);
  const [raceName, setRaceName] = useState("");
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
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
      try {
        const v = await api.modelsVersions();
        if (!cancelled) setVersions(v.versions);
      } catch {
        /* optional */
      }
      try {
        const c = await api.circuits();
        if (!cancelled) setCircuits(c.circuits);
      } catch {
        /* circuit metadata optional — Advanced mode still works */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!circuits?.length || !schema || raceName) return;
    const c = circuits.find((x) => x.race_name === "Italian Grand Prix") ?? circuits[0];
    setRaceName(c.race_name);
    setValues((prev) => ({
      ...prev,
      round_number: c.round,
      total_laps: c.total_laps,
      year: UI_YEAR,
      race_name: c.race_name,
    }));
  }, [circuits, schema, raceName, setValues]);

  /** After model switch, `useSchemaForm` resets — re-apply circuit round/laps from the selected race. */
  useEffect(() => {
    if (!circuits?.length || !schema || !raceName) return;
    const c = circuits.find((x) => x.race_name === raceName) ?? circuits[0];
    setValues((prev) => ({
      ...prev,
      round_number: c.round,
      total_laps: c.total_laps,
      year: UI_YEAR,
      race_name: c.race_name,
    }));
  }, [circuits, schema?.model_version, raceName, setValues]);

  const mergedInputs = useMemo(
    () => ({
      ...values,
      race_name: (values.race_name as string) || raceName || "Italian Grand Prix",
      year: UI_YEAR,
    }),
    [values, raceName],
  );

  /** Drop readonly model inputs so the API runs `build_single_row` and recomputes derived speeds/progress. */
  const sanitizedInputs = useMemo(() => {
    const ro = new Set(schema?.features.filter((f) => f.readonly).map((f) => f.name) ?? []);
    const out: Record<string, unknown> = { ...mergedInputs };
    for (const k of ro) delete out[k];
    return out;
  }, [mergedInputs, schema?.features]);

  const onRaceChange = useCallback((name: string, _meta: CircuitMeta) => {
    setRaceName(name);
  }, []);

  const handleModelChange = useCallback(
    async (v: string) => {
      if (!schema || v === schema.model_version) return;
      setSwitchingModel(true);
      setErr(null);
      try {
        await api.modelsSwitch(v);
        const s = await api.modelsSchema();
        setSchema(s);
        setPred(null);
        setSens(null);
        setSensFeature("");
      } catch (e) {
        setErr(String(e));
      } finally {
        setSwitchingModel(false);
      }
    },
    [schema],
  );

  const onPredict = useCallback(async () => {
    setErr(null);
    setPredicting(true);
    try {
      const body = { inputs: sanitizedInputs, include_impacts: true, include_row: false };
      const r = await api.predictSingle(body);
      setPred(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setPredicting(false);
    }
  }, [sanitizedInputs]);

  const onSensitivity = useCallback(async () => {
    if (!sensFeature) return;
    setErr(null);
    try {
      const r = await api.sensitivity({
        inputs: sanitizedInputs,
        feature: sensFeature,
        min: undefined,
        max: undefined,
        steps: 32,
      });
      setSens(r);
    } catch (e) {
      setErr(String(e));
    }
  }, [sanitizedInputs, sensFeature]);

  const trained = useMemo(() => new Set(schema?.trained_feature_names ?? []), [schema?.trained_feature_names]);
  const numericFeatures: FeatureSchemaItem[] = useMemo(
    () =>
      schema?.features.filter(
        (f) => f.kind === "number" && (trained.size === 0 || trained.has(f.name)),
      ) ?? [],
    [schema?.features, trained],
  );

  const currentSensValue = sensFeature ? Number(values[sensFeature] ?? 0) : undefined;

  if (loading) return <div className="text-f1-muted">Loading schema…</div>;
  if (err && !schema) return <div className="text-red-400">{err}</div>;

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Battle prediction</h1>
          <p className="mt-2 max-w-2xl text-f1-muted">
            Pick a 2025 circuit, tune the battle, then run a prediction. Switch the <strong>model</strong>{" "}
            anytime. Use <strong>Advanced</strong> for full feature control and local sensitivity.
          </p>
        </div>
        {schema && (
          <div className="shrink-0">
            <ModelSwitcher
              versions={versions}
              active={schema.model_version}
              disabled={switchingModel || predicting}
              onChange={handleModelChange}
            />
            {switchingModel && <p className="mt-1 text-xs text-f1-muted">Loading model…</p>}
          </div>
        )}
      </div>

      {schema && (
        <ConstructorContext
          attackerTeam={String(values.attacker_team ?? "")}
          defenderTeam={String(values.defender_team ?? "")}
          year={UI_YEAR}
        />
      )}

      {schema && (
        <BattleForm
          features={schema.features}
          values={values}
          onChange={setValues}
          circuits={circuits}
          raceName={raceName || "Italian Grand Prix"}
          onRaceChange={onRaceChange}
          uiYear={schema.ui_year ?? UI_YEAR}
        />
      )}

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm">{err}</div>}

      <div className="flex justify-center">
        <button
          type="button"
          onClick={onPredict}
          disabled={predicting || switchingModel}
          className="min-w-[200px] rounded-xl bg-f1-red px-8 py-4 text-lg font-bold text-white shadow-xl transition hover:brightness-110 disabled:opacity-50"
        >
          {predicting ? "Scoring…" : "Predict overtake"}
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
                    {f.label || f.name}
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
              currentValue={Number.isFinite(currentSensValue) ? currentSensValue : undefined}
            />
          )}
        </div>
      )}
    </div>
  );
}
