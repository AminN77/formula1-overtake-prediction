import { TEAM_COLORS } from "../../styles/f1-theme";

export function TeamBadge({ name }: { name: string }) {
  const c = TEAM_COLORS[name] || "#8b949e";
  return (
    <span
      className="inline-flex items-center rounded px-2 py-0.5 text-xs font-bold text-white"
      style={{ backgroundColor: c }}
    >
      {name}
    </span>
  );
}
