import { useEffect, useState } from "react";
import { api } from "../api/client";

export function ModelInfo() {
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const m = await api.modelsCurrent();
        setInfo(m);
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, []);

  if (err) return <div className="text-red-400">{err}</div>;
  if (!info) return <div className="text-f1-muted">Loading…</div>;

  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Active model</h1>
      <div className="rounded-xl border border-white/10 bg-f1-surface/50 p-4 font-mono text-sm text-f1-muted">
        <pre className="whitespace-pre-wrap">{JSON.stringify(info, null, 2)}</pre>
      </div>
    </div>
  );
}
