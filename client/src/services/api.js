/**
 * Centralized API Service
 * =======================
 * Manages all HTTP communication with the FastAPI backend.
 * Uses VITE_API_BASE_URL (defaults to '' for Vite proxy, or http://localhost:8000).
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Generic fetch wrapper with standardized JSON response and error handling.
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const errorMsg = data?.detail || data?.error || `HTTP error ${response.status}`;
      const err = new Error(errorMsg);
      err.status = response.status;
      err.data = data;
      throw err;
    }

    return data;
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      const connErr = new Error('Unable to connect to the risk analysis service. Please check if backend is running.');
      connErr.status = 0;
      throw connErr;
    }
    throw err;
  }
}

// ----------------------------------------------------------------------
// Health
// ----------------------------------------------------------------------

export async function getHealth() {
  return request('/health');
}

// ----------------------------------------------------------------------
// Module 1: EDA / Analytics
// ----------------------------------------------------------------------

export async function getEDASummary() {
  return request('/api/eda/summary');
}

export async function getEDAInsights() {
  return request('/api/eda/insights');
}

// ----------------------------------------------------------------------
// Module 2: ML Inference & SHAP Explainability
// ----------------------------------------------------------------------

export async function getApplicantRisk(applicantId) {
  return request(`/api/ml/applicant/${applicantId}`);
}

export async function getApplicantExplanation(applicantId, topK = 5) {
  return request(`/api/ml/applicant/${applicantId}/explanation?top_k=${topK}`);
}

// ----------------------------------------------------------------------
// Module 3: Talk-to-Data Chatbot
// ----------------------------------------------------------------------

export async function sendChatMessage(message, sessionId = null) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: message.trim(),
      session_id: sessionId,
    }),
  });
}
