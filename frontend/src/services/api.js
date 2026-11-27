/**
 * API Service Layer — Connects React Frontend to Flask ML Backend
 * Uses relative `/api` paths (proxied by Vite during dev) or `VITE_API_BASE_URL`.
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

class ApiError extends Error {
  constructor(message, status, payload = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
    this.fieldErrors = payload?.field_errors || null;
  }
}

async function requestJson(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  let response;

  try {
    response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });
  } catch {
    throw new ApiError(
      'Unable to reach the ML backend service. If deployed on a free tier, the server may be waking up—please retry in a few seconds.',
      0
    );
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.message ||
      `Request failed with HTTP status ${response.status}.`;
    throw new ApiError(message, response.status, data);
  }

  return data;
}

export function fetchHealth() {
  return requestJson('/api/health', { method: 'GET' });
}

export function fetchModelInfo(disease = 'diabetes') {
  return requestJson(`/api/model-info?disease=${encodeURIComponent(disease)}`, {
    method: 'GET',
  });
}

export function fetchModelMetrics(disease = 'diabetes') {
  return requestJson(`/api/model-metrics?disease=${encodeURIComponent(disease)}`, {
    method: 'GET',
  });
}

export function fetchSampleInputs() {
  return requestJson('/api/sample-inputs', { method: 'GET' });
}

export function submitPrediction(inputPayload) {
  return requestJson('/api/predict', {
    method: 'POST',
    body: JSON.stringify(inputPayload),
  });
}

export function getFigureUrl(filename) {
  return `${API_BASE}/api/figures/${encodeURIComponent(filename)}`;
}

export { ApiError };
