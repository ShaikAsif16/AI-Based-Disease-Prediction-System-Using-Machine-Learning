import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  FileCheck2,
  GraduationCap,
  Layers,
  Lock,
  ShieldCheck,
  Sparkles,
  Stethoscope,
} from 'lucide-react';
import { fetchModelInfo, fetchModelMetrics } from '../services/api';
import DisclaimerBanner from '../components/DisclaimerBanner';

const HOW_IT_WORKS = [
  {
    step: '01',
    title: 'Validated Clinical Input',
    description:
      'Enter 8 standardized clinical measurements with unit hints and strict range validation. Missing values are never fabricated.',
    icon: Stethoscope,
  },
  {
    step: '02',
    title: 'Leakage-Free Preprocessing',
    description:
      'Inputs flow through the exact fitted scikit-learn pipeline (median imputer + standard scaler trained strictly on training folds).',
    icon: Layers,
  },
  {
    step: '03',
    title: 'Real ML Model Inference',
    description:
      'The selected trained classifier evaluates non-linear feature interactions in real time via the Flask REST API.',
    icon: BrainCircuit,
  },
  {
    step: '04',
    title: 'Cautious Risk Guidance',
    description:
      'Receive predicted class, estimated probability, reference-range context, limitations, and physician follow-up guidance.',
    icon: ShieldCheck,
  },
];

export default function HomePage() {
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [infoRes, metricsRes] = await Promise.all([
          fetchModelInfo('diabetes'),
          fetchModelMetrics('diabetes'),
        ]);
        if (active) {
          setModelInfo(infoRes);
          setMetrics(metricsRes);
          setApiError(null);
        }
      } catch (err) {
        if (active) {
          setApiError(err.message);
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => {
      active = false;
    };
  }, []);

  const selectedModel =
    metrics && metrics.selected_model_key
      ? metrics.models?.[metrics.selected_model_key]
      : null;

  return (
    <div className="space-y-20 pb-8">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-hero-mesh clinical-grid-bg border-b border-slate-200/70">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8 lg:py-24">
          <div className="grid items-center gap-12 lg:grid-cols-12">
            {/* Left Copy */}
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55 }}
              className="lg:col-span-7 space-y-6"
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-white/90 px-3.5 py-1.5 text-xs font-semibold text-navy-800 shadow-sm">
                <Sparkles className="h-3.5 w-3.5 text-cyan-600" />
                <span>AIDL Machine Learning Clinical Risk Screening Platform</span>
              </div>

              <h1 className="text-4xl font-extrabold tracking-tight text-navy-950 sm:text-5xl lg:text-[3.35rem] lg:leading-[1.08]">
                AI-Based Disease{' '}
                <span className="gradient-text">Prediction System</span>
              </h1>

              <p className="max-w-2xl text-base leading-relaxed text-slate-600 sm:text-lg">
                Machine learning-powered health risk assessment built on a genuine clinical dataset,
                leakage-free preprocessing pipeline, and multi-algorithm validation. Designed for
                transparent diabetes risk screening today, with a modular architecture ready for
                additional diseases.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <Link to="/predict" className="btn-primary text-base px-7 py-3.5">
                  Start Prediction
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link to="/analytics" className="btn-secondary text-base px-6 py-3.5">
                  <BarChart3 className="h-4 w-4 text-blue-600" />
                  Explore Model Analytics
                </Link>
              </div>

              <div className="flex flex-wrap items-center gap-x-6 gap-y-2 pt-3 text-xs font-medium text-slate-500">
                <span className="inline-flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  Real Scikit-Learn Pipeline
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <Database className="h-4 w-4 text-blue-600" />
                  Pima Indians Diabetes Dataset (N = 768)
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <Lock className="h-4 w-4 text-cyan-600" />
                  Zero Patient Data Stored
                </span>
              </div>
            </motion.div>

            {/* Right Futuristic Live Model Card */}
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="lg:col-span-5"
            >
              <div className="glass-navy relative overflow-hidden p-6 sm:p-7">
                <div className="pointer-events-none absolute -right-12 -top-12 h-44 w-44 rounded-full bg-cyan-500/20 blur-3xl" />
                <div className="pointer-events-none absolute -bottom-12 -left-12 h-44 w-44 rounded-full bg-blue-600/20 blur-3xl" />

                <div className="relative space-y-5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-400 ring-1 ring-cyan-400/30">
                        <Cpu className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-cyan-300">
                          Active Deployed Pipeline
                        </p>
                        <p className="text-base font-bold text-white">
                          {loading
                            ? 'Loading trained model…'
                            : selectedModel?.display_name || 'Trained ML Pipeline'}
                        </p>
                      </div>
                    </div>
                    <span className="rounded-full bg-emerald-500/20 px-2.5 py-1 text-[11px] font-semibold text-emerald-300 ring-1 ring-emerald-400/30">
                      Live Artifact
                    </span>
                  </div>

                  {apiError ? (
                    <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4 text-xs text-amber-100">
                      {apiError}
                    </div>
                  ) : selectedModel ? (
                    <>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                          <p className="text-[11px] font-medium text-slate-300">
                            Test Recall (Sensitivity)
                          </p>
                          <p className="mt-1 font-mono text-2xl font-extrabold text-cyan-300">
                            {(selectedModel.test_metrics.recall * 100).toFixed(1)}%
                          </p>
                          <p className="mt-0.5 text-[10px] text-slate-400">
                            CV Mean: {(selectedModel.cv_metrics.recall * 100).toFixed(1)}%
                          </p>
                        </div>

                        <div className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                          <p className="text-[11px] font-medium text-slate-300">Test ROC-AUC</p>
                          <p className="mt-1 font-mono text-2xl font-extrabold text-blue-300">
                            {selectedModel.test_metrics.roc_auc.toFixed(3)}
                          </p>
                          <p className="mt-0.5 text-[10px] text-slate-400">
                            CV Mean: {selectedModel.cv_metrics.roc_auc.toFixed(3)}
                          </p>
                        </div>

                        <div className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                          <p className="text-[11px] font-medium text-slate-300">Test F1-Score</p>
                          <p className="mt-1 font-mono text-xl font-bold text-white">
                            {(selectedModel.test_metrics.f1_score * 100).toFixed(1)}%
                          </p>
                        </div>

                        <div className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                          <p className="text-[11px] font-medium text-slate-300">Test Accuracy</p>
                          <p className="mt-1 font-mono text-xl font-bold text-white">
                            {(selectedModel.test_metrics.accuracy * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-white/5 p-3.5 text-xs text-slate-300">
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span>Stratified Train / Test Split</span>
                          <span className="font-mono text-cyan-300">
                            {metrics.split_summary.train_samples} / {metrics.split_summary.test_samples}
                          </span>
                        </div>
                        <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
                          <span>Algorithms Benchmarked</span>
                          <span className="font-mono text-white">
                            {Object.keys(metrics.models).length} Models (5-Fold CV)
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="grid grid-cols-2 gap-3 animate-pulse">
                      {[1, 2, 3, 4].map((n) => (
                        <div key={n} className="h-20 rounded-xl bg-white/10" />
                      ))}
                    </div>
                  )}

                  <div className="flex items-center justify-between border-t border-white/10 pt-3 text-xs text-slate-300">
                    <span className="flex items-center gap-1.5">
                      <Activity className="h-3.5 w-3.5 text-cyan-400" />
                      All metrics sourced live from backend
                    </span>
                    <Link
                      to="/analytics"
                      className="font-semibold text-cyan-300 hover:text-cyan-200"
                    >
                      Details →
                    </Link>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Main Content Container */}
      <div className="mx-auto max-w-7xl space-y-20 px-4 sm:px-6 lg:px-8">
        <DisclaimerBanner />

        {/* How It Works */}
        <section>
          <div className="max-w-2xl">
            <p className="text-xs font-bold uppercase tracking-widest text-blue-600">
              Clinical Inference Workflow
            </p>
            <h2 className="mt-2 text-2xl font-extrabold tracking-tight text-navy-950 sm:text-3xl">
              How the Prediction System Works
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              Every risk assessment passes through a reproducible, leakage-free machine learning
              pipeline from raw clinical inputs to interpretable risk estimation.
            </p>
          </div>

          <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {HOW_IT_WORKS.map((item, idx) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: idx * 0.08 }}
                  className="glass-card group p-6 transition-all duration-200 hover:-translate-y-1 hover:border-cyan-300"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50 text-blue-700 ring-1 ring-blue-100 group-hover:from-navy-900 group-hover:to-blue-800 group-hover:text-white transition-colors">
                      <Icon className="h-5 w-5" />
                    </div>
                    <span className="font-mono text-xs font-bold text-slate-400">
                      STEP {item.step}
                    </span>
                  </div>
                  <h3 className="mt-5 text-base font-bold text-navy-900">{item.title}</h3>
                  <p className="mt-2 text-xs leading-relaxed text-slate-600">{item.description}</p>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* Machine Learning Overview & Performance Preview */}
        <section className="grid gap-8 lg:grid-cols-12">
          <div className="glass-card p-7 lg:col-span-6 space-y-5">
            <div className="inline-flex items-center gap-2 rounded-lg bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700">
              <BrainCircuit className="h-3.5 w-3.5" />
              Machine Learning Overview
            </div>
            <h2 className="text-2xl font-extrabold text-navy-950">
              Rigorous Validation Over Accuracy Alone
            </h2>
            <p className="text-sm leading-relaxed text-slate-600">
              In medical risk screening, relying on accuracy alone can conceal high false-negative
              rates due to class imbalance (65.1% non-diabetic vs. 34.9% diabetic in the Pima
              cohort). Our pipeline trains four distinct algorithms using balanced class weights,
              evaluates them with 5-fold stratified cross-validation, and selects the optimal model
              by prioritizing <strong className="text-navy-900">Recall (Sensitivity)</strong>,{' '}
              <strong className="text-navy-900">ROC-AUC</strong>, and{' '}
              <strong className="text-navy-900">F1-Score</strong>.
            </p>

            <ul className="space-y-3 text-sm text-slate-700">
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-600" />
                <span>
                  <strong>Leakage Prevention:</strong> Stratified 80/20 split is performed prior to
                  handling invalid clinical zeros, median imputation, and feature scaling.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-600" />
                <span>
                  <strong>4 Algorithms Benchmarked:</strong> Logistic Regression, Decision Tree,
                  Random Forest, and Calibrated RBF-SVM with fixed seed (<code className="font-mono text-xs">42</code>).
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-600" />
                <span>
                  <strong>Multi-Disease Ready:</strong> Registry-driven backend architecture enables
                  plugging in Cardiovascular or Chronic Kidney Disease pipelines seamlessly.
                </span>
              </li>
            </ul>
          </div>

          {/* Model Performance Preview */}
          <div className="glass-card p-7 lg:col-span-6 flex flex-col justify-between space-y-5">
            <div>
              <div className="flex items-center justify-between">
                <div className="inline-flex items-center gap-2 rounded-lg bg-cyan-50 px-3 py-1 text-xs font-bold text-cyan-800">
                  <BarChart3 className="h-3.5 w-3.5" />
                  Model Performance Preview
                </div>
                <Link
                  to="/analytics"
                  className="text-xs font-semibold text-blue-600 hover:text-blue-800"
                >
                  Full Analytics →
                </Link>
              </div>
              <h2 className="mt-3 text-2xl font-extrabold text-navy-950">
                Hold-Out Test Benchmark (N = 154)
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Genuine test-set results loaded live from <code className="font-mono">/api/model-metrics</code>
              </p>
            </div>

            {metrics?.models ? (
              <div className="space-y-3">
                {Object.values(metrics.models).map((m) => {
                  const isSelected = m.key === metrics.selected_model_key;
                  return (
                    <div
                      key={m.key}
                      className={`rounded-xl border p-3.5 transition ${
                        isSelected
                          ? 'border-cyan-300 bg-gradient-to-r from-blue-50/80 to-cyan-50/70'
                          : 'border-slate-200/80 bg-slate-50/60'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-navy-900">{m.display_name}</span>
                          {isSelected && (
                            <span className="rounded-full bg-navy-900 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-cyan-300">
                              Selected
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 font-mono text-xs">
                          <span className="text-slate-600">
                            Rec:{' '}
                            <strong className="text-navy-900">
                              {(m.test_metrics.recall * 100).toFixed(1)}%
                            </strong>
                          </span>
                          <span className="text-slate-600">
                            AUC:{' '}
                            <strong className="text-blue-700">
                              {m.test_metrics.roc_auc.toFixed(3)}
                            </strong>
                          </span>
                          <span className="text-slate-600">
                            Acc:{' '}
                            <strong className="text-slate-800">
                              {(m.test_metrics.accuracy * 100).toFixed(1)}%
                            </strong>
                          </span>
                        </div>
                      </div>
                      {/* Visual Recall bar */}
                      <div className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-slate-200/80">
                        <div
                          className={`h-full rounded-full ${
                            isSelected
                              ? 'bg-gradient-to-r from-blue-600 to-cyan-500'
                              : 'bg-slate-400'
                          }`}
                          style={{ width: `${Math.round(m.test_metrics.recall * 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">
                {loading
                  ? 'Fetching verified model evaluation metrics…'
                  : 'Start the Flask backend to display live benchmark metrics.'}
              </div>
            )}
          </div>
        </section>

        {/* Student & Academic Project Card */}
        <section className="glass-navy relative overflow-hidden p-8 sm:p-10">
          <div className="pointer-events-none absolute right-0 top-0 h-64 w-64 rounded-full bg-cyan-500/15 blur-3xl" />
          <div className="relative grid items-center gap-8 lg:grid-cols-12">
            <div className="lg:col-span-8 space-y-3">
              <div className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs font-semibold text-cyan-300">
                <GraduationCap className="h-4 w-4" />
                Academic Project Submission
              </div>
              <h2 className="text-2xl font-extrabold text-white sm:text-3xl">
                AI-Based Disease Prediction System Using Machine Learning
              </h2>
              <p className="text-sm leading-relaxed text-slate-300">
                Developed as a full-stack Artificial Intelligence &amp; Deep Learning (AIDL)
                engineering project demonstrating end-to-end medical data preprocessing, supervised
                model comparison, sensitivity-first model selection, RESTful inference, and modern
                clinical UI design.
              </p>
            </div>

            <div className="lg:col-span-4 rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3 text-sm">
              <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
                <span className="text-slate-400">Student Name</span>
                <span className="font-bold text-white">Shaik Asif</span>
              </div>
              <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
                <span className="text-slate-400">Hall Ticket No.</span>
                <span className="font-mono font-bold text-cyan-300">2403A52220</span>
              </div>
              <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
                <span className="text-slate-400">Subject</span>
                <span className="font-semibold text-white">AIDL</span>
              </div>
              <div className="pt-1">
                <Link
                  to="/about"
                  className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-xs font-bold text-navy-900 transition hover:bg-cyan-50"
                >
                  <FileCheck2 className="h-4 w-4 text-blue-600" />
                  Read Complete Project Documentation
                </Link>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
