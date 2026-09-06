import React from 'react';
import { ShieldCheck, AlertTriangle, XCircle, CheckCircle2, User, Percent, Gauge, CheckSquare } from 'lucide-react';
import RiskGauge from './RiskGauge';

export default function RiskResult({ riskData }) {
  if (!riskData) return null;

  const {
    applicant_id,
    default_probability,
    risk_score,
    risk_band,
    prediction,
    decision,
  } = riskData;

  const decisionBadgeConfig = {
    Low: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      text: 'text-emerald-400',
      icon: CheckCircle2,
      recommendation: 'Approve / Low Risk',
    },
    Medium: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      icon: AlertTriangle,
      recommendation: 'Manual Review',
    },
    High: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      text: 'text-rose-400',
      icon: XCircle,
      recommendation: 'High Risk / Reject Recommendation',
    },
  };

  const badge = decisionBadgeConfig[risk_band] || decisionBadgeConfig.Low;
  const DecisionIcon = badge.icon;

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
      {/* Top Banner with Decision Recommendation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="p-3 rounded-xl bg-slate-800 border border-slate-700 text-sky-400">
            <User className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-mono text-slate-400">APPLICANT PROFILE</div>
            <h2 className="text-2xl font-bold text-white font-mono tracking-tight">#{applicant_id}</h2>
          </div>
        </div>

        <div className={`flex items-center space-x-2.5 px-4 py-2.5 rounded-xl border ${badge.bg} ${badge.border} ${badge.text}`}>
          <DecisionIcon className="w-5 h-5 shrink-0" />
          <div className="text-right">
            <div className="text-[10px] font-semibold uppercase tracking-wider opacity-80">Decision Support</div>
            <div className="text-sm font-bold tracking-tight">{decision}</div>
          </div>
        </div>
      </div>

      {/* Grid of Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Default Probability */}
        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-medium mb-1">
            <Percent className="w-4 h-4 text-sky-400" />
            <span>Default Probability</span>
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {(default_probability * 100).toFixed(2)}%
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Calibrated ML risk estimate</div>
        </div>

        {/* Risk Score */}
        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-medium mb-1">
            <Gauge className="w-4 h-4 text-sky-400" />
            <span>Risk Score</span>
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {risk_score.toFixed(2)} <span className="text-xs font-normal text-slate-400">/ 100</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Standardized risk scale</div>
        </div>

        {/* Risk Tier & Binary Target */}
        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-medium mb-1">
            <CheckSquare className="w-4 h-4 text-sky-400" />
            <span>Risk Tier & Class</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`text-xl font-bold ${badge.text}`}>{risk_band.toUpperCase()}</span>
            <span className="text-xs text-slate-400 font-mono">(Class {prediction})</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            {prediction === 0 ? 'Non-Default classification' : 'Default classification'}
          </div>
        </div>
      </div>

      {/* Visual Risk Gauge Meter */}
      <div className="pt-2">
        <RiskGauge score={risk_score} riskBand={risk_band} probability={default_probability} />
      </div>
    </div>
  );
}
