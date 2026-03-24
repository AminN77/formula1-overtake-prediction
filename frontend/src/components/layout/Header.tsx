import { Link, useLocation } from "react-router-dom";

const nav = [
  { to: "/", label: "Battle" },
  { to: "/batch", label: "Batch" },
  { to: "/model", label: "Model" },
];

export function Header() {
  const loc = useLocation();
  return (
    <header className="border-b border-white/10 bg-f1-surface/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-3">
          <img src="/f1-logo.svg" alt="" className="h-9 w-auto" width={120} height={32} />
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-f1-muted">Formula 1</div>
            <div className="text-lg font-bold tracking-tight">Overtake Predictor</div>
          </div>
        </Link>
        <nav className="flex gap-1">
          {nav.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              className={`rounded px-3 py-2 text-sm font-semibold transition ${
                loc.pathname === n.to
                  ? "bg-f1-red text-white"
                  : "text-f1-muted hover:bg-white/5 hover:text-white"
              }`}
            >
              {n.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
