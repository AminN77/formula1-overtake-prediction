import { useAppData } from "../context/AppDataContext";

export function ModelInfo() {
  const { currentModel, versions, error, switchingModel, switchModel } = useAppData();

  if (error && !currentModel) return <div className="text-red-400">{error}</div>;
  if (!currentModel) return <div className="text-f1-muted">Loading…</div>;

  const active = String(currentModel.version ?? "");

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Active model</h1>

      {versions.length > 0 && (
        <div className="rounded-xl border border-white/10 bg-f1-surface/50 p-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-f1-red">Switch version</h2>
          <p className="mt-1 text-sm text-f1-muted">
            Loads a different artifact from the registry. The Battle page uses the same active model.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {versions.map((v) => (
              <button
                key={v}
                type="button"
                disabled={switchingModel}
                onClick={() => void switchModel(v)}
                className={`rounded-lg px-4 py-2 font-mono text-sm font-semibold transition ${
                  v === active ? "bg-f1-red text-white" : "bg-white/10 text-f1-muted hover:bg-white/15 hover:text-white"
                } disabled:opacity-50`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{error}</div>}

      <div className="rounded-xl border border-white/10 bg-f1-surface/50 p-4 font-mono text-sm text-f1-muted">
        <pre className="whitespace-pre-wrap">{JSON.stringify(currentModel, null, 2)}</pre>
      </div>
    </div>
  );
}
