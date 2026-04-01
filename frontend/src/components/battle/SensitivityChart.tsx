import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function SensitivityChart({
  feature,
  curve,
  baseline,
  currentValue,
}: {
  feature: string;
  curve: { value: number; probability: number }[];
  baseline: number;
  currentValue?: number;
}) {
  return (
    <div className="h-72 w-full rounded-xl border border-white/10 bg-f1-surface/40 p-4">
      <div className="mb-2 text-sm font-semibold text-white">
        Sensitivity: <span className="text-f1-red">{feature}</span>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={curve} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="value"
            stroke="#8b949e"
            fontSize={11}
            tickFormatter={(x: number) => (typeof x === "number" ? x.toFixed(5) : String(x))}
          />
          <YAxis
            domain={[0, 1]}
            stroke="#8b949e"
            fontSize={11}
            tickFormatter={(x: number) => (typeof x === "number" ? x.toFixed(5) : String(x))}
          />
          <Tooltip
            contentStyle={{ background: "#1e1e2e", border: "1px solid #333" }}
            formatter={(v: number, name: string) => [v.toFixed(5), name === "probability" ? "P" : name]}
          />
          <ReferenceLine y={baseline} stroke="#666" strokeDasharray="4 4" />
          {currentValue !== undefined && (
            <ReferenceLine x={currentValue} stroke="#e10600" strokeDasharray="4 4" />
          )}
          <Line type="monotone" dataKey="probability" stroke="#e10600" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
