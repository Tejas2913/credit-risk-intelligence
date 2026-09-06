import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import { Info, TrendingUp, BarChart2 } from 'lucide-react';

const BAR_COLORS = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185', '#34d399'];

export default function InsightChart({ insight, isFullWidth = false }) {
  if (!insight || !insight.chart) return null;

  const { title, metric, chart, interpretation, id } = insight;

  // Transform labels and values into Recharts data array
  const data = chart.labels.map((label, index) => ({
    name: label,
    value: chart.values[index],
    count: chart.counts ? chart.counts[index] : null,
  }));

  const isPercentage = id !== 1; // Chart 1 is absolute counts; 2-6 are percentages

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const pData = payload[0].payload;
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl text-xs space-y-1">
          <div className="font-semibold text-slate-200">{label}</div>
          <div className="text-sky-400 font-mono text-sm">
            {isPercentage ? `${pData.value.toFixed(1)}% default rate` : `${pData.value.toLocaleString()} applicants`}
          </div>
          {pData.count !== null && (
            <div className="text-slate-400">
              Sample size: <span className="text-slate-300 font-mono">{pData.count.toLocaleString()}</span>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`glass-panel p-5 rounded-xl border border-slate-800 flex flex-col justify-between ${isFullWidth ? 'col-span-full' : ''}`}>
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                Insight #{id}
              </span>
              <h3 className="text-base font-semibold text-white tracking-tight">{title}</h3>
            </div>
            {metric && (
              <p className="mt-1 text-xs text-sky-400 font-medium flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5" />
                {metric}
              </p>
            )}
          </div>
        </div>

        {/* Chart Visualization */}
        <div className="h-64 w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="name"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                interval={0}
                angle={-15}
                textAnchor="end"
                height={40}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(val) => (isPercentage ? `${val}%` : `${(val / 1000).toFixed(0)}k`)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={id === 1 ? (index === 0 ? '#10b981' : '#f43f5e') : BAR_COLORS[index % BAR_COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Business Interpretation */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-start gap-2 text-xs text-slate-300 leading-relaxed bg-slate-900/40 p-3 rounded-lg border border-slate-800/60">
        <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-200">Business Interpretation: </span>
          <span>{interpretation}</span>
        </div>
      </div>
    </div>
  );
}
