import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';

export default function ConfusionMatrixCard({ modelData }) {
  if (!modelData?.confusion_matrix) return null;

  const { tn, fp, fn, tp } = modelData.confusion_matrix;
  const total = tn + fp + fn + tp;
  const sensitivity = tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) : '0.0';
  const specificity = tn + fp > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) : '0.0';

  const cells = [
    {
      label: 'True Negative (TN)',
      sublabel: 'Actual: Non-Diabetic · Pred: Non-Diabetic',
      value: tn,
      pct: ((tn / total) * 100).toFixed(1),
      tone: 'bg-emerald-50 border-emerald-200 text-emerald-900',
      badge: 'Correctly Cleared',
    },
    {
      label: 'False Positive (FP)',
      sublabel: 'Actual: Non-Diabetic · Pred: Diabetic',
      value: fp,
      pct: ((fp / total) * 100).toFixed(1),
      tone: 'bg-amber-50 border-amber-200 text-amber-900',
      badge: 'Type I Error (Follow-Up Flag)',
    },
    {
      label: 'False Negative (FN)',
      sublabel: 'Actual: Diabetic · Pred: Non-Diabetic',
      value: fn,
      pct: ((fn / total) * 100).toFixed(1),
      tone: 'bg-rose-50 border-rose-200 text-rose-900',
      badge: 'Type II Error (Missed Risk)',
    },
    {
      label: 'True Positive (TP)',
      sublabel: 'Actual: Diabetic · Pred: Diabetic',
      value: tp,
      pct: ((tp / total) * 100).toFixed(1),
      tone: 'bg-cyan-50 border-cyan-200 text-navy-900',
      badge: 'Correctly Detected',
    },
  ];

  return (
    <div className="glass-card p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-lg font-bold text-navy-900">
            Confusion Matrix — {modelData.display_name}
          </h3>
          <p className="text-xs text-slate-500">
            Evaluated on the stratified hold-out test set (N = {total} unseen patients)
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-semibold">
          <span className="rounded-lg bg-blue-50 px-2.5 py-1 text-blue-800">
            Sensitivity: {sensitivity}%
          </span>
          <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-slate-700">
            Specificity: {specificity}%
          </span>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {cells.map((cell) => (
          <div
            key={cell.label}
            className={`rounded-xl border p-4 transition-transform duration-150 hover:-translate-y-0.5 ${cell.tone}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wide">{cell.label}</span>
              <span className="rounded-md bg-white/80 px-2 py-0.5 text-[11px] font-semibold">
                {cell.pct}%
              </span>
            </div>
            <p className="mt-2 font-mono text-3xl font-extrabold">{cell.value}</p>
            <p className="mt-1 text-[11px] opacity-80">{cell.sublabel}</p>
            <p className="mt-2 text-xs font-semibold">{cell.badge}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-slate-50 p-3.5 text-xs text-slate-600">
        {fn <= 15 ? (
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
        ) : (
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        )}
        <p className="leading-relaxed">
          <strong className="text-navy-900">Clinical Screening Interpretation:</strong> Out of{' '}
          <span className="font-mono font-semibold">{tp + fn}</span> actual positive diabetes cases
          in the hold-out test set, {modelData.display_name} detected{' '}
          <span className="font-mono font-semibold">{tp}</span> cases and missed{' '}
          <span className="font-mono font-semibold">{fn}</span> (False Negatives). Minimizing False
          Negatives is critical in preventive health screening.
        </p>
      </div>
    </div>
  );
}
