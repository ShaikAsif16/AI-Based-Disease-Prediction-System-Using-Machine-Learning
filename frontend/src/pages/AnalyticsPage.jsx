import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  Award,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Database,
  Image as ImageIcon,
  Layers,
  Loader2,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchModelInfo, fetchModelMetrics, getFigureUrl } from '../services/api';
import ConfusionMatrixCard from '../components/ConfusionMatrixCard';

const MODEL_COLORS = {
  logistic_regression: '#0F2942',
  decision_tree: '#94A3B8',
  random_forest: '#2563EB',
  svm: '#06B6D4',
};

const FIGURE_LABELS = {
  'model_comparison.png': 'Hold-Out Test Model Comparison',
  'confusion_matrices.png': 'Confusion Matrices (All 4 Algorithms)',
  'roc_curves.png': 'ROC Curves Comparison',
  'eda_class_and_zeros.png': 'Class Balance & Invalid Zero Analysis',
  'eda_feature_distributions.png': 'Feature Distributions by Outcome',
  'eda_correlation_heatmap.png': 'Clinical Feature Correlation Matrix',
};

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [activeModelKey, setActiveModelKey] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [metricsRes, infoRes] = await Promise.all([
          fetchModelMetrics('diabetes'),
          fetchModelInfo('diabetes'),
        ]);
        if (!active) return;
        setMetrics(metricsRes);
        setModelInfo(infoRes);
        setActiveModelKey(metricsRes.selected_model_key || Object.keys(metricsRes.models)[0]);
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => {
      active = false;
    };
  }, []);

  const modelList = useMemo(() => {
    if (!metrics?.models) return [];
    return Object.values(metrics.models);
  }, [metrics]);

  const selectedModel = metrics?.models?.[metrics.selected_model_key] || null;
  const inspectedModel = metrics?.models?.[activeModelKey] || selectedModel;

  // Grouped comparison chart data
  const comparisonChartData = useMemo(() => {
    if (!modelList.length) return [];
    const metricKeys = [
      { key: 'accuracy', label: 'Accuracy' },
      { key: 'precision', label: 'Precision' },
      { key: 'recall', label: 'Recall (Sens.)' },
      { key: 'f1_score', label: 'F1-Score' },
      { key: 'roc_auc', label: 'ROC-AUC' },
    ];
    return metricKeys.map(({ key, label }) => {
      const row = { metric: label };
      for (const m of modelList) {
        row[m.key] = Number((m.test_metrics[key] * 100).toFixed(2));
      }
      return row;
    });
  }, [modelList]);

  // Invalid zeros chart data
  const invalidZeroData = useMemo(() => {
    const counts = metrics?.dataset_summary?.invalid_zero_counts;
    const pcts = metrics?.dataset_summary?.invalid_zero_percentages;
    if (!counts) return [];
    return Object.entries(counts)
      .map(([feature, count]) => ({
        feature,
        count,
        pct: pcts?.[feature] ?? 0,
      }))
      .sort((a, b) => b.count - a.count);
  }, [metrics]);

  // Class balance chart data
  const classDistData = useMemo(() => {
    const dist = metrics?.dataset_summary?.class_distribution;
    if (!dist) return [];
    return [
      { name: 'Non-Diabetic (0)', count: dist.negative, pct: dist.negative_percentage, fill: '#0F2942' },
      { name: 'Diabetic (1)', count: dist.positive, pct: dist.positive_percentage, fill: '#06B6D4' },
    ];
  }, [metrics]);

  if (loading) {
    return (
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-center gap-3 px-4 py-28 text-slate-600">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        <p className="text-sm font-semibold">Loading verified ML evaluation metrics from backend…</p>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-20">
        <div className="glass-card border-rose-200 bg-rose-50/70 p-8 text-center space-y-3">
          <AlertCircle className="mx-auto h-10 w-10 text-rose-600" />
          <h1 className="text-xl font-bold text-navy-950">Unable to Load Model Analytics</h1>
          <p className="text-sm text-rose-800">{error || 'No evaluation data returned.'}</p>
        </div>
      </div>
    );
  }

  const kpis = selectedModel
    ? [
        {
          label: 'Recall (Sensitivity)',
          value: `${(selectedModel.test_metrics.recall * 100).toFixed(1)}%`,
          sub: `5-Fold CV: ${(selectedModel.cv_metrics.recall * 100).toFixed(1)}%`,
          accent: 'text-cyan-600',
        },
        {
          label: 'ROC-AUC Score',
          value: selectedModel.test_metrics.roc_auc.toFixed(4),
          sub: `5-Fold CV: ${selectedModel.cv_metrics.roc_auc.toFixed(4)}`,
          accent: 'text-blue-700',
        },
        {
          label: 'F1-Score',
          value: `${(selectedModel.test_metrics.f1_score * 100).toFixed(1)}%`,
          sub: `5-Fold CV: ${(selectedModel.cv_metrics.f1_score * 100).toFixed(1)}%`,
          accent: 'text-navy-900',
        },
        {
          label: 'Accuracy',
          value: `${(selectedModel.test_metrics.accuracy * 100).toFixed(1)}%`,
          sub: `5-Fold CV: ${(selectedModel.cv_metrics.accuracy * 100).toFixed(1)}%`,
          accent: 'text-navy-900',
        },
        {
          label: 'Precision',
          value: `${(selectedModel.test_metrics.precision * 100).toFixed(1)}%`,
          sub: `Specificity: ${(selectedModel.test_metrics.specificity * 100).toFixed(1)}%`,
          accent: 'text-slate-800',
        },
      ]
    : [];

  const diagonalData = [
    { fpr: 0, tpr: 0 },
    { fpr: 1, tpr: 1 },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-10 px-4 py-10 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-bold text-cyan-800">
            <BarChart3 className="h-3.5 w-3.5" />
            Genuine Training &amp; Validation Telemetry
          </div>
          <h1 className="mt-3 text-3xl font-extrabold tracking-tight text-navy-950 sm:text-4xl">
            Model Evaluation &amp; Analytics
          </h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            All metrics, confusion matrices, ROC curves, and dataset summaries below are computed
            directly from our reproducible scikit-learn training pipeline (seed{' '}
            <code className="font-mono text-xs">{metrics.split_summary.random_state}</code>) on the
            Pima Indians Diabetes Database.
          </p>
        </div>

        <div className="glass-card px-4 py-2.5 text-xs text-slate-600">
          <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Dataset Split (Stratified)
          </span>
          <span className="font-mono font-bold text-navy-900">
            {metrics.split_summary.train_samples} Train · {metrics.split_summary.test_samples} Hold-Out Test
          </span>
        </div>
      </div>

      {/* Selected Model Banner + KPIs */}
      {selectedModel && (
        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-navy p-6 sm:p-8 space-y-6"
        >
          <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-start">
            <div className="space-y-2 max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-bold text-cyan-300 ring-1 ring-cyan-400/30">
                <Award className="h-3.5 w-3.5" />
                Selected Deployment Model
              </div>
              <h2 className="text-2xl font-extrabold text-white sm:text-3xl">
                {selectedModel.display_name}{' '}
                <span className="text-base font-medium text-slate-300">
                  ({selectedModel.family})
                </span>
              </h2>
              <p className="text-sm leading-relaxed text-slate-300">
                {metrics.selection_rationale}
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-5 py-4 text-right shrink-0">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-cyan-300">
                CV Clinical Screening Score
              </p>
              <p className="mt-1 font-mono text-3xl font-extrabold text-white">
                {selectedModel.cv_metrics.clinical_selection_score.toFixed(4)}
              </p>
              <p className="mt-0.5 font-mono text-[10px] text-slate-400">
                0.45·Rec + 0.35·AUC + 0.20·F1
              </p>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {kpis.map((kpi) => (
              <div
                key={kpi.label}
                className="rounded-2xl border border-white/10 bg-white p-4 text-navy-900 shadow-sm"
              >
                <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  {kpi.label}
                </p>
                <p className={`mt-1.5 font-mono text-2xl font-extrabold ${kpi.accent}`}>
                  {kpi.value}
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">{kpi.sub}</p>
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* Algorithm Comparison Bar Chart */}
      <section className="glass-card p-6 sm:p-7 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold text-navy-950">
              Algorithm Performance Comparison (Hold-Out Test Set)
            </h2>
            <p className="text-xs text-slate-500">
              Percentage comparison across Accuracy, Precision, Recall (Sensitivity), F1-Score, and
              ROC-AUC on 154 unseen stratified test patients.
            </p>
          </div>
        </div>

        <div className="h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonChartData} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="metric" tick={{ fill: '#334155', fontSize: 12, fontWeight: 600 }} />
              <YAxis
                domain={[0, 100]}
                unit="%"
                tick={{ fill: '#64748B', fontSize: 11 }}
              />
              <Tooltip
                formatter={(val) => [`${val}%`, '']}
                contentStyle={{
                  borderRadius: '12px',
                  borderColor: '#CBD5E1',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
              {modelList.map((m) => (
                <Bar
                  key={m.key}
                  dataKey={m.key}
                  name={m.display_name + (m.key === metrics.selected_model_key ? ' ★' : '')}
                  fill={MODEL_COLORS[m.key] || '#2563EB'}
                  radius={[6, 6, 0, 0]}
                  maxBarSize={28}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed Comparison Table */}
        <div className="overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-left text-xs sm:text-sm">
            <thead className="bg-slate-50 text-[11px] font-bold uppercase tracking-wider text-slate-600">
              <tr>
                <th className="px-4 py-3">Algorithm</th>
                <th className="px-4 py-3">CV Score</th>
                <th className="px-4 py-3">Test Accuracy</th>
                <th className="px-4 py-3">Test Precision</th>
                <th className="px-4 py-3">Test Recall</th>
                <th className="px-4 py-3">Test F1</th>
                <th className="px-4 py-3">Test ROC-AUC</th>
                <th className="px-4 py-3">Missed Cases (FN)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {modelList.map((m) => {
                const isSelected = m.key === metrics.selected_model_key;
                return (
                  <tr
                    key={m.key}
                    className={isSelected ? 'bg-cyan-50/50 font-semibold' : 'hover:bg-slate-50'}
                  >
                    <td className="px-4 py-3 text-navy-900">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: MODEL_COLORS[m.key] }}
                        />
                        <span>{m.display_name}</span>
                        {isSelected && (
                          <span className="rounded-full bg-navy-900 px-2 py-0.5 text-[10px] font-bold text-cyan-300">
                            Selected
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono font-bold text-blue-700">
                      {m.cv_metrics.clinical_selection_score.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 font-mono">{(m.test_metrics.accuracy * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono">{(m.test_metrics.precision * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono font-bold text-cyan-700">
                      {(m.test_metrics.recall * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 font-mono">{(m.test_metrics.f1_score * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono font-bold text-navy-900">
                      {m.test_metrics.roc_auc.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 font-mono text-rose-700">
                      {m.confusion_matrix.fn} / {m.confusion_matrix.tp + m.confusion_matrix.fn}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ROC Curve + Interactive Model Inspector */}
      <div className="grid gap-8 lg:grid-cols-12">
        {/* ROC Curve */}
        <section className="glass-card p-6 lg:col-span-6 space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-blue-600" />
            <div>
              <h2 className="text-lg font-bold text-navy-950">ROC Curves (Hold-Out Test Set)</h2>
              <p className="text-xs text-slate-500">
                True Positive Rate (Sensitivity) vs. False Positive Rate across classification thresholds
              </p>
            </div>
          </div>

          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 10, right: 16, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  allowDuplicatedCategory={false}
                  label={{ value: 'False Positive Rate (1 − Specificity)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#475569' }}
                  tick={{ fontSize: 11, fill: '#64748B' }}
                />
                <YAxis
                  dataKey="tpr"
                  type="number"
                  domain={[0, 1]}
                  label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#475569' }}
                  tick={{ fontSize: 11, fill: '#64748B' }}
                />
                <Tooltip
                  formatter={(val) => Number(val).toFixed(3)}
                  labelFormatter={(lbl) => `FPR: ${Number(lbl).toFixed(3)}`}
                  contentStyle={{ borderRadius: '12px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />

                <Line
                  data={diagonalData}
                  dataKey="tpr"
                  name="Random Baseline (AUC = 0.500)"
                  stroke="#CBD5E1"
                  strokeDasharray="6 6"
                  dot={false}
                  strokeWidth={1.5}
                />

                {modelList.map((m) => (
                  <Line
                    key={m.key}
                    data={m.roc_curve}
                    dataKey="tpr"
                    name={`${m.short_name} (AUC = ${m.test_metrics.roc_auc.toFixed(3)})`}
                    stroke={MODEL_COLORS[m.key] || '#2563EB'}
                    strokeWidth={m.key === metrics.selected_model_key ? 3 : 1.8}
                    dot={false}
                    type="linear"
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Interactive Confusion Matrix Inspector */}
        <section className="lg:col-span-6 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {modelList.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => setActiveModelKey(m.key)}
                className={`rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                  activeModelKey === m.key
                    ? 'bg-navy-900 text-white shadow-sm'
                    : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                {m.short_name}
                {m.key === metrics.selected_model_key ? ' ★' : ''}
              </button>
            ))}
          </div>

          <ConfusionMatrixCard modelData={inspectedModel} />
        </section>
      </div>

      {/* Feature Importance (for selected model) */}
      {selectedModel?.feature_importance?.length > 0 && (
        <section className="glass-card p-6 sm:p-7 space-y-4">
          <div className="flex items-center gap-2">
            <BrainCircuit className="h-5 w-5 text-cyan-600" />
            <div>
              <h2 className="text-lg font-bold text-navy-950">
                Relative Feature Importance — {selectedModel.display_name}
              </h2>
              <p className="text-xs text-slate-500">
                Normalized Gini importance computed strictly from the trained Random Forest ensemble
              </p>
            </div>
          </div>

          <div className="h-[290px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={selectedModel.feature_importance.map((fi) => ({
                  ...fi,
                  pct: Number((fi.importance * 100).toFixed(2)),
                }))}
                layout="vertical"
                margin={{ top: 4, right: 32, left: 40, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" unit="%" tick={{ fontSize: 11, fill: '#64748B' }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={150}
                  tick={{ fontSize: 12, fill: '#0F2942', fontWeight: 600 }}
                />
                <Tooltip
                  formatter={(v) => [`${v}%`, 'Relative Importance']}
                  contentStyle={{ borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="pct" fill="#2563EB" radius={[0, 8, 8, 0]} barSize={22}>
                  {selectedModel.feature_importance.map((entry, idx) => (
                    <Cell
                      key={entry.feature}
                      fill={idx === 0 ? '#0B192C' : idx === 1 ? '#2563EB' : '#06B6D4'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* Dataset Summary & Invalid Zero Detection */}
      <section className="grid gap-8 lg:grid-cols-12">
        <div className="glass-card p-6 lg:col-span-5 space-y-4">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-blue-600" />
            <div>
              <h2 className="text-lg font-bold text-navy-950">Dataset Class Distribution</h2>
              <p className="text-xs text-slate-500">
                {metrics.dataset_summary.dataset_name} (N = {metrics.dataset_summary.total_samples})
              </p>
            </div>
          </div>

          <div className="h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classDistData} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#0F2942', fontWeight: 600 }} />
                <YAxis tick={{ fontSize: 11, fill: '#64748B' }} />
                <Tooltip
                  formatter={(val, _n, props) => [`${val} patients (${props.payload.pct}%)`, 'Count']}
                  contentStyle={{ borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" radius={[8, 8, 0, 0]} maxBarSize={64}>
                  {classDistData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="text-xs leading-relaxed text-slate-600">
            Because positive cases represent{' '}
            <strong>{metrics.dataset_summary.class_distribution.positive_percentage}%</strong> of
            records, we apply stratified splitting and balanced class weighting so classifiers do not
            bias toward the majority negative class.
          </p>
        </div>

        <div className="glass-card p-6 lg:col-span-7 space-y-4">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-cyan-600" />
            <div>
              <h2 className="text-lg font-bold text-navy-950">
                Physiologically Invalid Zero Values Handled
              </h2>
              <p className="text-xs text-slate-500">
                Zeros in clinical fields below denote unrecorded measurements and are imputed inside
                training folds only
              </p>
            </div>
          </div>

          <div className="h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={invalidZeroData}
                layout="vertical"
                margin={{ top: 4, right: 28, left: 24, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748B' }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={110}
                  tick={{ fontSize: 12, fill: '#0F2942', fontWeight: 600 }}
                />
                <Tooltip
                  formatter={(val, _n, props) => [
                    `${val} zeros (${props.payload.pct}% of dataset)`,
                    'Invalid Zeros',
                  ]}
                  contentStyle={{ borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" fill="#2563EB" radius={[0, 8, 8, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {modelInfo?.training_configuration?.training_fitted_medians && (
            <div className="rounded-xl bg-slate-50 p-3.5">
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Training-Only Fitted Imputation Medians (Zero Data Leakage)
              </p>
              <div className="mt-2 flex flex-wrap gap-2 font-mono text-xs">
                {Object.entries(modelInfo.training_configuration.training_fitted_medians).map(
                  ([col, val]) => (
                    <span
                      key={col}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-navy-900"
                    >
                      {col}: <strong>{val}</strong>
                    </span>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Methodology & Leakage Prevention */}
      <section className="glass-card p-6 sm:p-8 space-y-5">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-emerald-600" />
          <h2 className="text-xl font-bold text-navy-950">
            Validation &amp; Leakage Prevention Methodology
          </h2>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-5 space-y-2">
            <span className="font-mono text-xs font-bold text-blue-700">01 · SPLIT BEFORE FIT</span>
            <h3 className="text-sm font-bold text-navy-900">Stratified 80/20 Partition</h3>
            <p className="text-xs leading-relaxed text-slate-600">
              Data is partitioned with <code className="font-mono">stratify=y</code> and{' '}
              <code className="font-mono">random_state=42</code> before computing any medians or
              scaling parameters, preserving the exact 34.9% positive class ratio in both splits.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-5 space-y-2">
            <span className="font-mono text-xs font-bold text-cyan-700">02 · PIPELINE ISOLATION</span>
            <h3 className="text-sm font-bold text-navy-900">In-Fold Preprocessing</h3>
            <p className="text-xs leading-relaxed text-slate-600">
              Our custom <code className="font-mono">InvalidZeroToNaNTransformer</code>,{' '}
              <code className="font-mono">SimpleImputer(median)</code>, and{' '}
              <code className="font-mono">StandardScaler</code> run inside a unified scikit-learn
              Pipeline, ensuring validation folds never leak statistics into training folds.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-5 space-y-2">
            <span className="font-mono text-xs font-bold text-emerald-700">03 · CLINICAL SELECTION</span>
            <h3 className="text-sm font-bold text-navy-900">Sensitivity-First Selection</h3>
            <p className="text-xs leading-relaxed text-slate-600">
              Model selection relies strictly on 5-fold cross-validation metrics on the training set
              (prioritizing Recall and ROC-AUC), keeping the 154-patient test set completely
              untouched until final reporting.
            </p>
          </div>
        </div>
      </section>

      {/* Generated Matplotlib / Seaborn Visualizations */}
      {Array.isArray(metrics.generated_figures) && metrics.generated_figures.length > 0 && (
        <section className="glass-card p-6 sm:p-8 space-y-5">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-5 w-5 text-blue-600" />
            <div>
              <h2 className="text-xl font-bold text-navy-950">
                Generated Matplotlib &amp; Seaborn Evaluation Figures
              </h2>
              <p className="text-xs text-slate-500">
                High-resolution static academic plots automatically produced by{' '}
                <code className="font-mono">backend/ml/train.py</code> and stored in{' '}
                <code className="font-mono">reports/figures/</code>
              </p>
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {metrics.generated_figures.map((filename) => (
              <figure
                key={filename}
                className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
              >
                <div className="border-b border-slate-100 bg-slate-50 px-4 py-2.5 flex items-center justify-between">
                  <figcaption className="text-xs font-bold text-navy-900">
                    {FIGURE_LABELS[filename] || filename}
                  </figcaption>
                  <span className="inline-flex items-center gap-1 font-mono text-[10px] text-emerald-700">
                    <CheckCircle2 className="h-3 w-3" />
                    {filename}
                  </span>
                </div>
                <div className="p-3 bg-white">
                  <img
                    src={getFigureUrl(filename)}
                    alt={FIGURE_LABELS[filename] || filename}
                    loading="lazy"
                    className="h-auto w-full rounded-lg object-contain"
                  />
                </div>
              </figure>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
