import React from 'react';

export default function RiskGauge({ score = 0, riskBand = 'Low', probability = 0 }) {
  // Clamp score between 0 and 100
  const clampedScore = Math.max(0, Math.min(100, score));

  // Determine indicator position percentage
  const positionPct = `${clampedScore}%`;

  const bandConfig = {
    Low: {
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      barColor: 'bg-emerald-500',
    },
    Medium: {
      color: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      barColor: 'bg-amber-500',
    },
    High: {
      color: 'text-rose-400',
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      barColor: 'bg-rose-500',
    },
  };

  const currentStyle = bandConfig[riskBand] || bandConfig.Low;

  return (
    <div className="w-full space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Risk Score Spectrum</span>
        <div className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono border ${currentStyle.bg} ${currentStyle.color} ${currentStyle.border}`}>
          {clampedScore.toFixed(2)} / 100
        </div>
      </div>

      {/* Segmented Risk Gauge Bar */}
      <div className="relative pt-6 pb-2">
        {/* Track */}
        <div className="h-3 w-full rounded-full bg-slate-800 flex overflow-hidden p-0.5 gap-1">
          {/* Low Tier (0 - 30%) */}
          <div className="w-[30%] bg-emerald-500/70 rounded-l-full relative group">
            <div className="absolute inset-0 bg-emerald-400/20 group-hover:bg-emerald-400/40 transition-colors" />
          </div>
          {/* Medium Tier (30 - 60%) */}
          <div className="w-[30%] bg-amber-500/70 relative group">
            <div className="absolute inset-0 bg-amber-400/20 group-hover:bg-amber-400/40 transition-colors" />
          </div>
          {/* High Tier (60 - 100%) */}
          <div className="w-[40%] bg-rose-500/70 rounded-r-full relative group">
            <div className="absolute inset-0 bg-rose-400/20 group-hover:bg-rose-400/40 transition-colors" />
          </div>
        </div>

        {/* Needle / Score Indicator Marker */}
        <div
          className="absolute top-1 transform -translate-x-1/2 flex flex-col items-center transition-all duration-500 ease-out"
          style={{ left: positionPct }}
        >
          <div className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono text-white shadow-lg ${currentStyle.barColor}`}>
            {clampedScore.toFixed(1)}
          </div>
          <div
            className={`w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[6px] ${
              riskBand === 'High' ? 'border-t-rose-500' : riskBand === 'Medium' ? 'border-t-amber-500' : 'border-t-emerald-500'
            }`}
          />
        </div>
      </div>

      {/* Tier Labels & Threshold Marks */}
      <div className="flex justify-between text-[11px] font-mono text-slate-400 px-1">
        <div className="flex flex-col items-start">
          <span className="text-emerald-400 font-semibold">LOW (0 - 30)</span>
          <span className="text-[10px] text-slate-400">Approve</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-amber-400 font-semibold">MEDIUM (30 - 60)</span>
          <span className="text-[10px] text-slate-400">Manual Review</span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-rose-400 font-semibold">HIGH (60 - 100)</span>
          <span className="text-[10px] text-slate-400">Reject / High Risk</span>
        </div>
      </div>
    </div>
  );
}
