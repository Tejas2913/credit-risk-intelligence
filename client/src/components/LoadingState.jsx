import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState({ message = 'Loading analytics data...' }) {
  return (
    <div className="glass-panel p-12 rounded-2xl border border-slate-800 flex flex-col items-center justify-center space-y-4 my-8">
      <div className="w-12 h-12 rounded-2xl bg-brand-500/10 border border-brand-500/30 flex items-center justify-center text-brand-400">
        <Loader2 className="w-6 h-6 animate-spin text-sky-400" />
      </div>
      <div className="text-center">
        <div className="text-sm font-semibold text-slate-200">{message}</div>
        <div className="text-xs text-slate-500 mt-1">Retrieving from analytical database & inference engine</div>
      </div>
    </div>
  );
}
