import { useId } from "react";

/** Small “i” chip; shows description on hover/focus (CSS-only popover). */
export function InfoTooltip({ text }: { text: string | null | undefined }) {
  const id = useId();
  if (!text) return null;
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        className="ml-1 inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-f1-muted/60 text-[10px] font-bold leading-none text-f1-muted transition hover:border-f1-red hover:text-white focus:outline-none focus:ring-1 focus:ring-f1-red"
        aria-describedby={id}
      >
        i
      </button>
      <span
        id={id}
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-1/2 z-50 mb-1 w-56 -translate-x-1/2 rounded-md border border-white/15 bg-f1-card px-2 py-1.5 text-left text-xs leading-snug text-f1-muted opacity-0 shadow-xl transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}
