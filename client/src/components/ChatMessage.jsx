import React, { useState } from 'react';
import { Bot, User, Code, Database, ChevronDown, ChevronUp, CheckCircle, AlertCircle, Sparkles, ExternalLink } from 'lucide-react';

export default function ChatMessage({ message, onNavigateToApplicant }) {
  const [showSql, setShowSql] = useState(false);
  const [showData, setShowData] = useState(false);

  const isUser = message.sender === 'user';
  const { text, sql, data = [], columns = [], intent, used_llm, error, success } = message;

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="flex items-start space-x-2.5 max-w-2xl">
          <div className="bg-brand-600 text-white px-4 py-3 rounded-2xl rounded-tr-none shadow-md text-sm leading-relaxed">
            {text}
          </div>
          <div className="w-8 h-8 rounded-full bg-brand-500/20 border border-brand-500/40 flex items-center justify-center text-brand-300 shrink-0">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>
    );
  }

  // Assistant Message
  return (
    <div className="flex justify-start mb-6">
      <div className="flex items-start space-x-3 max-w-3xl w-full">
        {/* Assistant Avatar */}
        <div className="w-8 h-8 rounded-full bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400 shrink-0 mt-0.5">
          <Bot className="w-4 h-4" />
        </div>

        {/* Message Content Container */}
        <div className="space-y-3 flex-1">
          <div className="glass-panel p-4 sm:p-5 rounded-2xl rounded-tl-none border border-slate-800 text-slate-200 text-sm leading-relaxed space-y-3">
            {/* Primary Business Answer */}
            <p className="whitespace-pre-line text-slate-100 font-normal">{text}</p>

            {/* Badges Bar (Intent, Engine) */}
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
              {intent && (
                <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Intent: {intent}
                </span>
              )}
              <span className={`px-2 py-0.5 rounded border ${used_llm ? 'bg-purple-500/10 text-purple-300 border-purple-500/30' : 'bg-sky-500/10 text-sky-300 border-sky-500/30'}`}>
                {used_llm ? 'Engine: LLM Grounded' : 'Engine: Deterministic Schema'}
              </span>
              {data && data.length > 0 && (
                <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                  {data.length} row{data.length > 1 ? 's' : ''} retrieved
                </span>
              )}
            </div>

            {/* Optional Expandable SQL Section */}
            {sql && (
              <div className="pt-1">
                <button
                  onClick={() => setShowSql(!showSql)}
                  className="flex items-center space-x-1.5 text-xs font-mono text-sky-400 hover:text-sky-300 transition-colors"
                >
                  <Code className="w-3.5 h-3.5" />
                  <span>{showSql ? 'Hide Generated SQL' : 'View Generated SQL Query'}</span>
                  {showSql ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {showSql && (
                  <div className="mt-2 p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-sky-300 overflow-x-auto">
                    <pre className="whitespace-pre-wrap">{sql}</pre>
                  </div>
                )}
              </div>
            )}

            {/* Optional Expandable Result Data Table */}
            {data && data.length > 0 && columns && columns.length > 0 && (
              <div>
                <button
                  onClick={() => setShowData(!showData)}
                  className="flex items-center space-x-1.5 text-xs font-mono text-sky-400 hover:text-sky-300 transition-colors"
                >
                  <Database className="w-3.5 h-3.5" />
                  <span>{showData ? 'Hide Raw Result Data' : `View Result Dataset (${data.length} rows)`}</span>
                  {showData ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {showData && (
                  <div className="mt-2 rounded-lg border border-slate-800 overflow-x-auto max-h-60">
                    <table className="w-full text-left text-xs text-slate-300 font-mono">
                      <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 sticky top-0">
                        <tr>
                          {columns.map((col, idx) => (
                            <th key={idx} className="px-3 py-2 uppercase font-semibold text-[11px] whitespace-nowrap">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-slate-950/80">
                        {data.map((row, rIdx) => (
                          <tr key={rIdx} className="hover:bg-slate-900/60">
                            {columns.map((col, cIdx) => {
                              const val = row[col];
                              const formatted = typeof val === 'number' ? (Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2)) : String(val ?? '');
                              return (
                                <td key={cIdx} className="px-3 py-1.5 whitespace-nowrap">
                                  {formatted}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
