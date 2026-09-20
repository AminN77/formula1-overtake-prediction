import { COMPOUND_COLORS } from "../../styles/f1-theme";

export function TyreIcon({ compound }: { compound: string }) {
  const c = COMPOUND_COLORS[compound.toUpperCase()] || "#888";
  return (
    <span className="inline-flex items-center gap-1 text-xs font-bold uppercase">
      <span className="h-3 w-3 rounded-full border border-white/20" style={{ background: c }} />
      {compound}
    </span>
  );
}
