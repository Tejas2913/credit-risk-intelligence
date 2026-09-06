import React from 'react';

export default function SummaryCard({ title, value, subtitle, icon: Icon, accent = 'brand' }) {
  const accentStyles = {
    brand: {
      border: 'border-brand-500/30 hover:border-brand-500/50',
      bg: 'bg-brand-500/10 text-brand-400',
      glow: 'shadow-brand-500/5',
    },
    emerald: {
      border: 'border-emerald-500/30 hover:border-emerald-500/50',
      bg: 'bg-emerald-500/10 text-emerald-400',
      glow: 'shadow-emerald-500/5',
    },
    rose: {
      border: 'border-rose-500/30 hover:border-rose-500/50',
      bg: 'bg-rose-500/10 text-rose-400',
      glow: 'shadow-rose-500/5',
    },
    amber: {
      border: 'border-amber-500/30 hover:border-amber-500/50',
      bg: 'bg-amber-500/10 text-amber-400',
      glow: 'shadow-amber-500/5',
    },
  };

  const style = accentStyles[accent] || accentStyles.brand;

  return (
    <div className={`glass-panel p-5 rounded-xl border transition-all duration-200 ${style.border} shadow-lg ${style.glow}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</span>
        {Icon && (
          <div className={`p-2.5 rounded-lg ${style.bg}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
      <div className="mt-3 flex items-baseline space-x-2">
        <span className="text-2xl sm:text-3xl font-bold tracking-tight text-white">{value}</span>
      </div>
      {subtitle && (
        <p className="mt-1.5 text-xs text-slate-400">{subtitle}</p>
      )}
    </div>
  );
}
