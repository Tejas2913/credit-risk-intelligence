import React, { useState, useEffect } from 'react';
import { Search, Activity, ShieldAlert, Sparkles, AlertCircle, ArrowRight, UserCheck, HelpCircle } from 'lucide-react';
import RiskResult from '../components/RiskResult';
import ShapFactors from '../components/ShapFactors';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { getApplicantRisk, getApplicantExplanation } from '../services/api';

const SAMPLE_APPLICANTS = [
  { id: 396899, label: '396899 (Benchmark: Low Risk)', band: 'Low' },
  { id: 100002, label: '100002 (High Risk)', band: 'High' },
  { id: 100007, label: '100007 (Medium Risk / Review)', band: 'Medium' },
  { id: 100003, label: '100003 (Low Risk)', band: 'Low' },
  { id: 100004, label: '100004 (Low Risk)', band: 'Low' },
];

export default function RiskAnalysis() {
  const [applicantIdInput, setApplicantIdInput] = useState('396899');
  const [activeApplicantId, setActiveApplicantId] = useState(396899);
  const [riskData, setRiskData] = useState(null);
  const [explanationData, setExplanationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeApplicant = async (idToAnalyze) => {
    const id = parseInt(idToAnalyze, 10);
    if (isNaN(id) || id <= 0) {
      setError('Please enter a valid numeric Applicant ID.');
      return;
    }

    setLoading(true);
    setError(null);
    setActiveApplicantId(id);

    try {
      // Parallel fetch for risk score and SHAP explanation
      const [riskRes, explanationRes] = await Promise.all([
        getApplicantRisk(id),
        getApplicantExplanation(id, 6),
      ]);

      setRiskData(riskRes);
      setExplanationData(explanationRes);
    } catch (err) {
      console.error('Error analyzing applicant:', err);
      if (err.status === 404) {
        setError('Applicant ID not found. Please check the ID and try again.');
      } else if (err.status === 0) {
        setError('Unable to connect to the risk analysis service. Please verify that the backend is running.');
      } else {
        setError(err.message || 'An error occurred while evaluating credit risk for this applicant.');
      }
      setRiskData(null);
      setExplanationData(null);
    } finally {
      setLoading(false);
    }
  };

  // Run initial benchmark analysis on component mount
  useEffect(() => {
    analyzeApplicant(396899);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    analyzeApplicant(applicantIdInput);
  };

  const handleSelectSample = (sampleId) => {
    setApplicantIdInput(sampleId.toString());
    analyzeApplicant(sampleId);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div>
        <div className="flex items-center space-x-2 text-sky-400 text-xs font-mono uppercase tracking-wider">
          <Activity className="w-4 h-4" />
          <span>Module 2: ML Inference & Explainability (XAI)</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mt-1">
          Individual Applicant Credit Risk Analysis
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Real-time CatBoost default risk scoring, risk band calibration, and TreeSHAP feature attributions.
        </p>
      </div>

      {/* Applicant ID Search & Sample Selector Card */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Search className="w-5 h-5" />
            </div>
            <input
              type="number"
              value={applicantIdInput}
              onChange={(e) => setApplicantIdInput(e.target.value)}
              placeholder="Enter SK_ID_CURR (e.g. 396899)..."
              disabled={loading}
              className="w-full bg-slate-900 border border-slate-700 rounded-xl pl-11 pr-4 py-3 text-sm text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition-all disabled:opacity-50"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !applicantIdInput}
            className="bg-brand-600 hover:bg-brand-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium px-6 py-3 rounded-xl transition-all shadow-md flex items-center justify-center space-x-2 shrink-0 disabled:cursor-not-allowed"
          >
            <Activity className="w-4 h-4" />
            <span>Analyze Risk</span>
          </button>
        </form>

        {/* Quick Sample Applicants */}
        <div className="flex items-center space-x-2 overflow-x-auto pt-1 text-xs text-slate-400 scrollbar-none">
          <span className="shrink-0 font-medium text-slate-400">Quick Samples:</span>
          {SAMPLE_APPLICANTS.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSelectSample(sample.id)}
              disabled={loading}
              className={`px-3 py-1 rounded-lg border text-xs font-mono transition-all shrink-0 ${
                activeApplicantId === sample.id
                  ? 'bg-brand-500/20 text-brand-300 border-brand-500/50'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-300 border-slate-800'
              }`}
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>

      {/* Loading State */}
      {loading && <LoadingState message={`Scoring applicant #${activeApplicantId} and calculating TreeSHAP values...`} />}

      {/* Error Message */}
      {error && !loading && (
        <ErrorMessage
          title="Applicant Risk Scoring Error"
          message={error}
          onRetry={() => analyzeApplicant(activeApplicantId)}
        />
      )}

      {/* Results Section */}
      {!loading && !error && riskData && (
        <div className="space-y-6">
          {/* Main Risk Result Card & Gauge */}
          <RiskResult riskData={riskData} />

          {/* SHAP Explainability Breakdown */}
          {explanationData && <ShapFactors explanation={explanationData} />}
        </div>
      )}

      {/* Bottom Mandatory Regulatory Disclaimer */}
      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80 text-xs text-slate-400 flex items-start space-x-2.5">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <span className="font-semibold text-slate-300">Decision-Support Notice: </span>
          This system provides AI-assisted credit risk analysis for decision support. It does not make autonomous lending decisions. All scoring recommendations are intended for qualified underwriting personnel review.
        </p>
      </div>
    </div>
  );
}
