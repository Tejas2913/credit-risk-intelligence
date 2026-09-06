import React, { useState } from 'react';
import { Send, Sparkles, Loader2 } from 'lucide-react';

const SUGGESTED_QUESTIONS = [
  'What is the default rate?',
  'Show default rate by education level',
  'What is the default rate for applicants with previous refusals?',
  'Compare default rates by income group',
  'How does late payment history relate to default risk?',
  'Which organization types have the highest default rates?',
  'What is the default rate for highly leveraged applicants?'
];

export default function ChatInput({ onSendMessage, isLoading }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleSelectSuggested = (q) => {
    setInput(q);
  };

  return (
    <div className="space-y-3">
      {/* Suggested Questions Chips */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1.5 scrollbar-none">
        <div className="flex items-center space-x-1 text-xs text-slate-400 shrink-0 font-medium">
          <Sparkles className="w-3.5 h-3.5 text-sky-400" />
          <span>Suggested:</span>
        </div>
        {SUGGESTED_QUESTIONS.map((question, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleSelectSuggested(question)}
            disabled={isLoading}
            className="text-xs px-3 py-1 rounded-full bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white border border-slate-700/80 transition-all shrink-0 disabled:opacity-50"
          >
            {question}
          </button>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the credit risk portfolio (e.g. 'What is the default rate by education?')..."
            disabled={isLoading}
            className="w-full bg-slate-900/90 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition-all disabled:opacity-50"
          />
        </div>

        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="bg-brand-600 hover:bg-brand-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium px-5 py-3 rounded-xl transition-all shadow-md flex items-center space-x-2 shrink-0 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="hidden sm:inline text-xs">Querying...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline text-xs">Send</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
