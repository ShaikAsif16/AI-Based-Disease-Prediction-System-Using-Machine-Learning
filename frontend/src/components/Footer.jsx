import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, GraduationCap, ShieldAlert } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-slate-200/80 bg-navy-950 text-slate-300">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-10 md:grid-cols-3">
          {/* Brand */}
          <div className="space-y-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 text-white">
                <Activity className="h-5 w-5" />
              </div>
              <div>
                <p className="text-base font-bold text-white">
                  MedRisk<span className="text-cyan-400">AI</span>
                </p>
                <p className="text-xs text-slate-400">AI-Based Disease Prediction System</p>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-slate-400">
              Full-stack clinical risk screening platform powered by a genuinely trained
              scikit-learn machine learning pipeline on the Pima Indians Diabetes Database.
              Zero patient data is stored by default.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400">
              Navigation
            </h3>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <Link to="/" className="text-slate-300 transition hover:text-white">
                  Home Dashboard
                </Link>
              </li>
              <li>
                <Link to="/predict" className="text-slate-300 transition hover:text-white">
                  Diabetes Risk Prediction
                </Link>
              </li>
              <li>
                <Link to="/analytics" className="text-slate-300 transition hover:text-white">
                  Model Evaluation &amp; Analytics
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-slate-300 transition hover:text-white">
                  About Academic Project
                </Link>
              </li>
            </ul>
          </div>

          {/* Student & Academic Details */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center gap-2 text-cyan-400">
              <GraduationCap className="h-4 w-4" />
              <span className="text-xs font-bold uppercase tracking-wider">
                AIDL Academic Project
              </span>
            </div>
            <dl className="mt-3 space-y-1.5 text-xs">
              <div className="flex justify-between gap-2">
                <dt className="text-slate-400">Student Name</dt>
                <dd className="font-semibold text-white">Shaik Asif</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-400">Hall Ticket No.</dt>
                <dd className="font-mono font-semibold text-cyan-300">2403A52220</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-400">Subject</dt>
                <dd className="font-semibold text-white">AI &amp; Deep Learning (AIDL)</dd>
              </div>
            </dl>
          </div>
        </div>

        {/* Educational Medical Disclaimer */}
        <div className="mt-10 flex flex-col gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-3.5 text-xs text-amber-100 sm:flex-row sm:items-center">
          <ShieldAlert className="h-4 w-4 shrink-0 text-amber-400" />
          <p className="leading-relaxed">
            <strong className="font-semibold text-white">Educational Screening Demonstration Only:</strong>{' '}
            This software does not provide medical advice, diagnosis, or treatment. Always consult a
            qualified healthcare professional regarding clinical measurements or health concerns.
          </p>
        </div>
      </div>
    </footer>
  );
}
