import React, { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Activity, BarChart3, BookOpen, Home, Menu, ShieldCheck, Stethoscope, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const NAV_ITEMS = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/predict', label: 'Disease Prediction', icon: Stethoscope },
  { to: '/analytics', label: 'Model Analytics', icon: BarChart3 },
  { to: '/about', label: 'About Project', icon: BookOpen },
];

export default function Navbar({ backendStatus }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const isHealthy = backendStatus?.status === 'healthy';

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/75 bg-white/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3.5 sm:px-6 lg:px-8">
        {/* Brand */}
        <Link to="/" className="group flex items-center gap-3 focus:outline-none">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-navy-900 via-blue-700 to-cyan-500 text-white shadow-md transition-transform duration-200 group-hover:scale-105">
            <Activity className="h-5 w-5" />
            <span className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-white bg-cyan-400" />
          </div>
          <div className="leading-tight">
            <span className="block text-sm font-extrabold tracking-tight text-navy-900 sm:text-base">
              MedRisk<span className="text-cyan-600">AI</span>
            </span>
            <span className="block text-[11px] font-medium text-slate-500">
              Disease Prediction System
            </span>
          </div>
        </Link>

        {/* Desktop Nav */}
        <nav aria-label="Main navigation" className="hidden items-center gap-1 md:flex">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-navy-900 text-white shadow-sm'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-navy-900'
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status Pill & Mobile Toggle */}
        <div className="flex items-center gap-3">
          <div
            title={
              isHealthy
                ? `Connected to Flask ML API (${backendStatus?.active_model || 'Trained Pipeline'})`
                : 'Flask ML backend status'
            }
            className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold sm:inline-flex ${
              isHealthy
                ? 'border-emerald-200 bg-emerald-50/90 text-emerald-800'
                : 'border-amber-200 bg-amber-50/90 text-amber-800'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
              }`}
            />
            <span>{isHealthy ? 'ML API Online' : 'Connecting API…'}</span>
          </div>

          <Link
            to="/predict"
            className="hidden items-center gap-1.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:brightness-105 lg:inline-flex"
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            Start Screening
          </Link>

          <button
            type="button"
            aria-label="Toggle navigation menu"
            aria-expanded={mobileOpen}
            onClick={() => setMobileOpen((prev) => !prev)}
            className="inline-flex items-center justify-center rounded-xl border border-slate-200 p-2 text-navy-900 hover:bg-slate-100 md:hidden"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.nav
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-slate-200 bg-white md:hidden"
          >
            <div className="space-y-1 px-4 py-3">
              {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-semibold transition ${
                      isActive
                        ? 'bg-navy-900 text-white'
                        : 'text-slate-700 hover:bg-slate-100'
                    }`
                  }
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </NavLink>
              ))}
              <div className="mt-2 flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-2 text-xs text-slate-600">
                <span>Backend Inference Engine</span>
                <span className={isHealthy ? 'font-semibold text-emerald-700' : 'font-semibold text-amber-700'}>
                  {isHealthy ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
