/** Dropdown to switch active model version (calls POST /api/models/switch). */
export function ModelSwitcher({
  versions,
  active,
  disabled,
  onChange,
}: {
  versions: string[];
  active: string;
  disabled?: boolean;
  onChange: (version: string) => void;
}) {
  if (!versions.length) {
    return (
      <span className="rounded-lg border border-white/10 bg-f1-card/80 px-3 py-2 font-mono text-xs text-f1-muted">
        Model: <span className="text-white">{active}</span>
      </span>
    );
  }
  return (
    <label className="flex flex-wrap items-center gap-2 text-sm">
      <span className="font-semibold text-white">Model</span>
      <select
        value={active}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="min-w-[5rem] rounded-lg border border-white/10 bg-f1-card px-3 py-2 font-mono text-sm text-white"
      >
        {versions.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </label>
  );
}
