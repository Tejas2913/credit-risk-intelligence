import React from 'react';
import { ArrowUpRight, ArrowDownRight, ShieldAlert, Sparkles, Scale, Info } from 'lucide-react';

export default function ShapFactors({ explanation }) {
  if (!explanation) return null;

  const {
    base_value,
    risk_increasing_factors = [],
    risk_reducing_factors = [],
    disclaimer,
  } = explanation;

  // Helper to format values cleanly
  const formatValue = (val) => {
    if (val === null || val === undefined || val === 'missing') return 'Missing / Not Recorded';
    if (typeof val === 'number') {
      if (Math.abs(val) >= 1000) return `$${val.toLocaleString()}`;
      if (Number.isInteger(val)) return val.toString();
      return val.toFixed(3);
    }
    return String(val);
  };

  // Helper to calculate relative contribution bar width
  const maxAbsShap = Math.max(
    ...risk_increasing_factors.map((f) => Math.abs(f.shap_value || 0)),
    ...risk_reducing_factors.map((f) => Math.abs(f.shap_value || 0)),
    0.01
  );

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-sky-400" />
            <h3 className="text-lg font-bold text-white tracking-tight">Explainable AI: Feature Attribution</h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            TreeSHAP feature contributions explaining why this applicant received this credit risk score.
          </p>
        </div>
        <div className="flex items-center space-x-2 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
          <Scale className="w-3.5 h-3.5 text-sky-400" />
          <span>Base Value: {base_value !== undefined ? base_value.toFixed(4) : 'N/A'}</span>
        </div>
      </div>

      {/* Two-Column Factor Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Increasing Factors */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2 text-sm font-semibold text-rose-400">
            <div className="p-1 rounded bg-rose-500/10 border border-rose-500/20">
              <ArrowUpRight className="w-4 h-4 text-rose-400" />
            </div>
            <span>Top Risk-Increasing Drivers (Pushes Score Higher)</span>
          </div>

          <div className="space-y-2.5">
            {risk_increasing_factors.length === 0 ? (
              <div className="text-xs text-slate-400 p-4 rounded-lg bg-slate-900/50 border border-slate-800 text-center">
                No significant risk-increasing factors identified.
              </div>
            ) : (
              risk_increasing_factors.map((factor, idx) => {
                const widthPct = Math.min(100, Math.round((Math.abs(factor.shap_value) / maxAbsShap) * 100));
                return (
                  <div key={idx} className="bg-slate-900/70 p-3 rounded-xl border border-slate-800/80 hover:border-rose-500/30 transition-all space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="text-xs font-semibold text-slate-200">
                          {factor.description || factor.feature}
                        </div>
                        <div className="text-[11px] font-mono text-slate-400">
                          Observed Value: <span className="text-slate-300 font-medium">{formatValue(factor.feature_value)}</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <span className="text-xs font-mono font-bold text-rose-400">
                          +{factor.shap_value.toFixed(4)}
                        </span>
                        <div className="text-[10px] text-rose-400/80 font-mono">↑ High Risk</div>
                      </div>
                    </div>
                    {/* Visual contribution bar */}
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-rose-500 rounded-full transition-all duration-500" style={{ width: `${widthPct}%` }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Risk Mitigating Factors */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2 text-sm font-semibold text-emerald-400">
            <div className="p-1 rounded bg-emerald-500/10 border border-emerald-500/20">
              <ArrowDownRight className="w-4 h-4 text-emerald-400" />
            </div>
            <span>Top Risk-Reducing Drivers (Pushes Score Lower)</span>
          </div>

          <div className="space-y-2.5">
            {risk_reducing_factors.length === 0 ? (
              <div className="text-xs text-slate-400 p-4 rounded-lg bg-slate-900/50 border border-slate-800 text-center">
                No significant risk-mitigating factors identified.
              </div>
            ) : (
              risk_reducing_factors.map((factor, idx) => {
                const widthPct = Math.min(100, Math.round((Math.abs(factor.shap_value) / maxAbsShap) * 100));
                return (
                  <div key={idx} className="bg-slate-900/70 p-3 rounded-xl border border-slate-800/80 hover:border-emerald-500/30 transition-all space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="text-xs font-semibold text-slate-200">
                          {factor.description || factor.feature}
                        </div>
                        <div className="text-[11px] font-mono text-slate-400">
                          Observed Value: <span className="text-slate-300 font-medium">{formatValue(factor.feature_value)}</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <span className="text-xs font-mono font-bold text-emerald-400">
                          {factor.shap_value.toFixed(4)}
                        </span>
                        <div className="text-[10px] text-emerald-400/80 font-mono">↓ Low Risk</div>
                      </div>
                    </div>
                    {/* Visual contribution bar */}
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 rounded-full transition-all duration-500" style={{ width: `${widthPct}%` }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Compliance & Regulatory Disclaimer */}
      <div className="pt-3 border-t border-slate-800/80 flex items-start gap-2.5 text-xs text-slate-400 bg-slate-900/40 p-3.5 rounded-xl border border-slate-800/60">
        <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <span className="font-semibold text-slate-300">Fair Lending Compliance: </span>
          {disclaimer || "Decision-support explanation only. Excludes protected demographic attributes (gender, family status, education, assets) in compliance with fair-lending standards."}
        </p>
      </div>
    </div>
  );
}
