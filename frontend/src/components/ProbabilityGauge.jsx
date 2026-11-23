import React from 'react';
import { motion } from 'framer-motion';

export default function ProbabilityGauge({ probability, riskLevel }) {
  const pct = Math.max(0, Math.min(100, Number((probability * 100).toFixed(1))));
  const radius = 84;
  const strokeWidth = 14;
  const circumference = Math.PI * radius; // Semi-circle arc length
  const dashOffset = circumference - (pct / 100) * circumference;

  const palette = {
    low: {
      stroke: '#059669',
      track: '#D1FAE5',
      badge: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    },
    moderate: {
      stroke: '#D97706',
      track: '#FEF3C7',
      badge: 'bg-amber-100 text-amber-900 border-amber-200',
    },
    elevated: {
      stroke: '#DC2626',
      track: '#FEE2E2',
      badge: 'bg-rose-100 text-rose-800 border-rose-200',
    },
  };

  const colors = palette[riskLevel] || palette.moderate;

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative h-36 w-56">
        <svg viewBox="0 0 220 130" className="h-full w-full overflow-visible">
          {/* Track */}
          <path
            d="M 26 110 A 84 84 0 0 1 194 110"
            fill="none"
            stroke={colors.track}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Animated Fill */}
          <motion.path
            d="M 26 110 A 84 84 0 0 1 194 110"
            fill="none"
            stroke={colors.stroke}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 1.0, ease: [0.22, 1, 0.36, 1] }}
          />
        </svg>

        <div className="absolute inset-x-0 bottom-2 flex flex-col items-center text-center">
          <span className="font-mono text-3xl font-extrabold tracking-tight text-navy-900">
            {pct}%
          </span>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Estimated Probability
          </span>
        </div>
      </div>

      <div className="mt-1 flex w-48 justify-between text-[11px] font-medium text-slate-400">
        <span>0% Low</span>
        <span>50%</span>
        <span>100% High</span>
      </div>
    </div>
  );
}
