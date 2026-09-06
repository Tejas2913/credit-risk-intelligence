import React, { useState, useEffect } from 'react';
import { Users, UserX, UserCheck, Percent, Layers, BarChart3, ShieldCheck } from 'lucide-react';
import SummaryCard from '../components/SummaryCard';
import InsightChart from '../components/InsightChart';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { getEDASummary, getEDAInsights } from '../services/api';

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [summaryRes, insightsRes] = await Promise.all([
        getEDASummary(),
        getEDAInsights(),
      ]);
      setSummary(summaryRes);
      setInsights(insightsRes.insights || []);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(err.message || 'Failed to communicate with analytics backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  if (loading) {
    return <LoadingState message="Loading portfolio analytics & exploratory data insights..." />;
  }

  if (error) {
    return (
      <ErrorMessage
        title="Failed to Load Portfolio Analytics"
        message={error}
        onRetry={fetchDashboardData}
      />
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-sky-400 text-xs font-mono uppercase tracking-wider">
            <BarChart3 className="w-4 h-4" />
            <span>Module 1: Exploratory Data Analysis</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mt-1">
            Portfolio Risk Overview & Insights
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Baseline distributions and historical risk behavioral relationships derived from the Home Credit analytical dataset.
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800">
          <Layers className="w-4 h-4 text-sky-400" />
          <span>Engineered Features: <strong className="text-white font-mono">{summary?.total_features || 231}</strong></span>
        </div>
      </div>

      {/* Top 4 Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <SummaryCard
          title="Total Evaluated Applicants"
          value={summary?.total_applicants ? summary.total_applicants.toLocaleString() : '307,511'}
          subtitle="Full baseline portfolio population"
          icon={Users}
          accent="brand"
        />
        <SummaryCard
          title="Default Applicants"
          value={summary?.default_applicants ? summary.default_applicants.toLocaleString() : '24,825'}
          subtitle="Observed loan default instances"
          icon={UserX}
          accent="rose"
        />
        <SummaryCard
          title="Non-Default Applicants"
          value={summary?.non_default_applicants ? summary.non_default_applicants.toLocaleString() : '282,686'}
          subtitle="Repaid / non-default obligations"
          icon={UserCheck}
          accent="emerald"
        />
        <SummaryCard
          title="Portfolio Default Rate"
          value={summary?.default_rate_pct ? `${summary.default_rate_pct.toFixed(2)}%` : '8.07%'}
          subtitle="Class imbalance ratio: ~11.4 to 1"
          icon={Percent}
          accent="amber"
        />
      </div>

      {/* 6 Core Business Insights Visualizations */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white tracking-tight">Core Business Insights (EDA)</h2>
          <span className="text-xs text-slate-400 font-mono">6 Authoritative Visualizations</span>
        </div>

        {/* Hero Insight: Distribution of Loan Default Outcomes */}
        {insights.length > 0 && (
          <div className="grid grid-cols-1 gap-6">
            <InsightChart insight={insights[0]} isFullWidth={true} />
          </div>
        )}

        {/* 2-Column Responsive Grid for Remaining 5 Insight Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {insights.slice(1).map((insight) => (
            <InsightChart key={insight.id} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
}
