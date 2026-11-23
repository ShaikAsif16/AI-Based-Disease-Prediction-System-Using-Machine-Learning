import React, { useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import PredictPage from './pages/PredictPage';
import ResultsPage from './pages/ResultsPage';
import AnalyticsPage from './pages/AnalyticsPage';
import AboutPage from './pages/AboutPage';
import { fetchHealth } from './services/api';

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [pathname]);
  return null;
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [latestResult, setLatestResult] = useState(null);

  useEffect(() => {
    let active = true;
    async function checkHealth() {
      try {
        const res = await fetchHealth();
        if (active) setBackendStatus(res);
      } catch {
        if (active) setBackendStatus({ status: 'offline' });
      }
    }
    checkHealth();
    const timer = setInterval(checkHealth, 30000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <BrowserRouter>
      <ScrollToTop />
      <div className="flex min-h-screen flex-col bg-clinical-bg text-navy-900">
        <Navbar backendStatus={backendStatus} />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route
              path="/predict"
              element={<PredictPage onPredictionComplete={setLatestResult} />}
            />
            <Route path="/results" element={<ResultsPage latestResult={latestResult} />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
