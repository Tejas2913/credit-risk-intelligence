import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import RiskAnalysis from './pages/RiskAnalysis';
import TalkToData from './pages/TalkToData';
import { getHealth } from './services/api';
import { ShieldCheck, Cpu, Database } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState('connecting');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await getHealth();
        if (res && res.status === 'ok') {
          setSystemStatus('ok');
        } else {
          setSystemStatus('degraded');
        }
      } catch (err) {
        setSystemStatus('error');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans text-slate-100">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemStatus={systemStatus}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'risk' && <RiskAnalysis />}
        {activeTab === 'chat' && <TalkToData />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2 text-slate-400">
            <ShieldCheck className="w-4 h-4 text-sky-400" />
            <span className="font-semibold text-slate-300">Credit Risk Intelligence Platform</span>
            <span>• Decision Support Engine</span>
          </div>
          <div className="flex items-center space-x-4 text-slate-500 text-[11px]">
            <span>Model: CatBoost Credit Risk</span>
            <span>•</span>
            <span>SHAP TreeExplainer</span>
            <span>•</span>
            <span>Talk-to-Data NL-to-SQL</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
