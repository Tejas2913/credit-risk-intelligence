import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Bot, Trash2, Sparkles, Database, ShieldCheck, HelpCircle } from 'lucide-react';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import { sendChatMessage } from '../services/api';

export default function TalkToData() {
  // Generate a persistent session identifier per browser tab session
  const [sessionId] = useState(() => `session_${Math.random().toString(36).substring(2, 9)}`);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'assistant',
      text: (
        "Welcome to Talk-to-Data! I can analyze the 307,511 applicant portfolio, generate schema-grounded SQL, " +
        "and synthesize executive business insights. How can I assist with your portfolio analysis today?"
      ),
      intent: 'welcome',
      used_llm: false,
      sql: null,
      data: [],
      columns: [],
      success: true,
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (text) => {
    const userMsg = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(text, sessionId);

      const assistantMsg = {
        id: `assistant_${Date.now()}`,
        sender: 'assistant',
        text: response.answer || 'No answer generated.',
        sql: response.sql,
        data: response.data || [],
        columns: response.columns || [],
        intent: response.intent,
        used_llm: response.used_llm,
        error: response.error,
        success: response.success,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg = {
        id: `err_${Date.now()}`,
        sender: 'assistant',
        text: err.message || 'An error occurred while communicating with the Talk-to-Data service.',
        sql: null,
        data: [],
        columns: [],
        intent: 'error',
        used_llm: false,
        error: err.message,
        success: false,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    setMessages([
      {
        id: 'welcome_reset',
        sender: 'assistant',
        text: "Conversation history cleared. You can start a new query about portfolio analytics.",
        intent: 'welcome',
        used_llm: false,
        sql: null,
        data: [],
        columns: [],
        success: true,
      },
    ]);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-sky-400 text-xs font-mono uppercase tracking-wider">
            <MessageSquare className="w-4 h-4" />
            <span>Module 3: Talk-to-Data Natural Language Query Engine</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mt-1">
            Talk to Your Credit Data
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Ask questions about the loan portfolio in natural language. Powered by schema-aware SQL generation and execution.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleClearHistory}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 text-xs transition-colors"
            title="Clear Chat History"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Chat</span>
          </button>
        </div>
      </div>

      {/* Main Chat Box Container */}
      <div className="glass-panel rounded-2xl border border-slate-800 flex flex-col h-[650px] shadow-2xl overflow-hidden">
        {/* Messages Scroll Area */}
        <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-4">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* Assistant Thinking / Querying Indicator */}
          {isLoading && (
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400 shrink-0">
                <Bot className="w-4 h-4 animate-pulse" />
              </div>
              <div className="glass-panel px-4 py-3 rounded-2xl rounded-tl-none border border-slate-800 text-slate-400 text-xs flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
                <span>Translating natural language → generating validated SQL query...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input & Suggested Chips Area */}
        <div className="p-4 sm:p-5 bg-slate-900/90 border-t border-slate-800/80">
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
}
