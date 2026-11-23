import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  HeartHandshake,
  Info,
  Lock,
  ShieldAlert,
  Stethoscope,
} from 'lucide-react';
import ProbabilityGauge from '../components/ProbabilityGauge';
import DisclaimerBanner from '../components/DisclaimerBanner';

const RISK_STYLES = {
  low: {
    badge: 'bg-emerald-100 text-emerald-900 border-emerald-200',
    panel: 'border-emerald-200 bg-gradient-to-br from-emerald-50/80 via-white to-cyan-50/40',
    dot: 'bg-emerald-500',
  },
  moderate: {
    badge: 'bg-amber-100 text-amber-950 border-amber-200',
    panel: 'border-amber-200 bg-gradient-to-br from-amber-50/80 via-white to-slate-50/50',
    dot: 'bg-amber-500',
  },
  elevated: {
    badge: 'bg-rose-100 text-rose-950 border-rose-200',
    panel: 'border-rose-200 bg-gradient-to-br from-rose-50/80 via-white to-amber-50/40',
    dot: 'bg-rose-500',
  },
};

const CONTEXT_BADGE = {
  within_reference: {
    label: 'Within Reference',
    cls: 'bg-emerald-100 text-emerald-800',
  },
  borderline: {
    label: 'Borderline',
    cls: 'bg-amber-100 text-amber-900',
  },
  elevated: {
    label: 'Elevated',
    cls: 'bg-rose-100 text-rose-800',
  },
  informational: {
    label: 'Context Factor',
    cls: 'bg-slate-100 text-slate-700',
  },
};

export default function ResultsPage({ latestResult }) {
  const location = useLocation();
  const result = location.state?.result || latestResult;

  if (!result) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-20 text-center sm:px-6">
        <div className="glass-card p-10 space-y-5">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-700">
            <Stethoscope className="h-7 w-7" />
          </div>
          <h1 className="text-2xl font-extrabold text-navy-950">
            No Prediction Result Available Yet
          </h1>
          <p className="text-sm text-slate-600">
            Submit your 8 clinical measurements on the Disease Prediction page to run a real
            inference through the trained machine learning model.
          </p>
          <div className="pt-2">
            <Link to="/predict" className="btn-primary">
              Go to Disease Prediction Form
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const theme = RISK_STYLES[result.risk_level] || RISK_STYLES.moderate;

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      {/* Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <Link
            to="/predict"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 hover:text-blue-900"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to Assessment Form
          </Link>
          <h1 className="mt-2 text-3xl font-extrabold tracking-tight text-navy-950">
            Risk Assessment Results
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Link to="/predict" className="btn-secondary">
            New Assessment
          </Link>
          <Link to="/analytics" className="btn-primary">
            <BarChart3 className="h-4 w-4" />
            View Model Analytics
          </Link>
        </div>
      </div>

      {/* Prominent Educational Disclaimer */}
      <DisclaimerBanner />

      {/* Primary Prediction Summary Card */}
      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className={`rounded-3xl border p-6 shadow-glass sm:p-8 ${theme.panel}`}
      >
        <div className="grid items-center gap-8 lg:grid-cols-12">
          <div className="space-y-4 lg:col-span-7">
            <div className="flex flex-wrap items-center gap-2.5">
              <span
                className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1 text-xs font-bold ${theme.badge}`}
              >
                <span className={`h-2 w-2 rounded-full ${theme.dot}`} />
                {result.risk_category}
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white/90 px-3 py-1 text-xs font-semibold text-slate-700">
                <Activity className="h-3.5 w-3.5 text-blue-600" />
                Model: {result.model_used}
              </span>
            </div>

            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Model Classification Output
              </p>
              <h2 className="mt-1 text-2xl font-extrabold text-navy-950 sm:text-3xl">
                {result.prediction_label}
              </h2>
            </div>

            <p className="text-sm leading-relaxed text-slate-700">{result.explanation}</p>

            <div className="flex flex-wrap items-center gap-4 pt-1 text-xs text-slate-600">
              <span className="inline-flex items-center gap-1.5 font-medium">
                <Lock className="h-3.5 w-3.5 text-emerald-600" />
                Data Stored on Server: <strong>No (In-Memory Only)</strong>
              </span>
              {result.class_probabilities && (
                <span className="font-mono text-slate-500">
                  P(Lower Risk) = {(result.class_probabilities.negative * 100).toFixed(1)}% ·
                  P(Higher Risk) = {(result.class_probabilities.positive * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-col items-center justify-center rounded-2xl border border-slate-200/70 bg-white/90 p-6 lg:col-span-5">
            {result.supports_probability && result.probability !== null ? (
              <ProbabilityGauge probability={result.probability} riskLevel={result.risk_level} />
            ) : (
              <div className="py-6 text-center text-sm text-slate-500">
                Probability estimates are not supported by this classifier configuration.
              </div>
            )}
            <p className="mt-3 text-center text-[11px] leading-relaxed text-slate-500">
              Statistical probability generated live by the trained{' '}
              <strong>{result.model_used}</strong> pipeline. Not a clinical diagnosis.
            </p>
          </div>
        </div>
      </motion.section>

      {/* Clinical Input Profile Context */}
      <section className="glass-card p-6 sm:p-7">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 pb-4">
          <div className="flex items-center gap-2.5">
            <ClipboardList className="h-5 w-5 text-blue-600" />
            <div>
              <h3 className="text-lg font-bold text-navy-900">
                Submitted Clinical Profile &amp; Reference Context
              </h3>
              <p className="text-xs text-slate-500">
                Informational comparison against general reference ranges (does not override ML output)
              </p>
            </div>
          </div>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {(result.clinical_context || []).map((item) => {
            const badge = CONTEXT_BADGE[item.status] || CONTEXT_BADGE.informational;
            return (
              <div
                key={item.feature}
                className="rounded-2xl border border-slate-200/80 bg-slate-50/60 p-4 space-y-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs font-bold text-navy-900">{item.label}</span>
                  <span className={`shrink-0 rounded-md px-2 py-0.5 text-[10px] font-bold ${badge.cls}`}>
                    {badge.label}
                  </span>
                </div>
                <p className="font-mono text-xl font-extrabold text-navy-950">
                  {item.value}{' '}
                  <span className="text-xs font-medium text-slate-500">{item.unit}</span>
                </p>
                <p className="text-[11px] leading-relaxed text-slate-600">{item.note}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Limitations & General Health Information */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Important Limitations */}
        <section className="glass-card p-6 sm:p-7 space-y-4">
          <div className="flex items-center gap-2.5 text-amber-800">
            <ShieldAlert className="h-5 w-5 text-amber-600" />
            <h3 className="text-lg font-bold text-navy-900">Important Model Limitations</h3>
          </div>
          <ul className="space-y-3 text-sm text-slate-600">
            {(result.limitations || []).map((item, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* General Health Information */}
        <section className="glass-card p-6 sm:p-7 space-y-4">
          <div className="flex items-center gap-2.5">
            <HeartHandshake className="h-5 w-5 text-cyan-600" />
            <h3 className="text-lg font-bold text-navy-900">General Preventive Health Information</h3>
          </div>
          <ul className="space-y-3 text-sm text-slate-600">
            {(result.general_health_information || []).map((item, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      {/* Professional Medical Recommendation */}
      <section className="glass-navy p-6 sm:p-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-400/30">
            <Info className="h-6 w-6" />
          </div>
          <div className="space-y-2">
            <h3 className="text-lg font-bold text-white">
              Consult a Qualified Healthcare Professional
            </h3>
            <p className="text-sm leading-relaxed text-slate-300">
              {result.professional_recommendation}
            </p>
            <p className="pt-1 text-xs text-cyan-200/90">{result.disclaimer}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
