import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertCircle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  FlaskConical,
  HeartPulse,
  Info,
  Loader2,
  Lock,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Stethoscope,
} from 'lucide-react';
import { fetchModelInfo, fetchSampleInputs, submitPrediction } from '../services/api';
import DisclaimerBanner from '../components/DisclaimerBanner';

const FALLBACK_FEATURE_SPECS = [
  {
    name: 'Pregnancies',
    label: 'Number of Pregnancies',
    unit: 'count',
    description: 'Total number of times pregnant (enter 0 if none).',
    clinical_reference: 'Typical range: 0–17',
    min_value: 0,
    max_value: 25,
    step: 1,
    allow_zero: true,
    integer_only: true,
    placeholder: 'e.g., 2',
  },
  {
    name: 'Glucose',
    label: 'Plasma Glucose Concentration',
    unit: 'mg/dL',
    description: 'Plasma glucose measured 2 hours after an oral glucose tolerance test (OGTT).',
    clinical_reference: 'Normal 2-hr OGTT: < 140 mg/dL | Prediabetes: 140–199 mg/dL',
    min_value: 40,
    max_value: 400,
    step: 1,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 118',
  },
  {
    name: 'BloodPressure',
    label: 'Diastolic Blood Pressure',
    unit: 'mm Hg',
    description: 'Resting diastolic blood pressure (bottom number in a blood pressure reading).',
    clinical_reference: 'Normal diastolic: 60–80 mm Hg',
    min_value: 24,
    max_value: 200,
    step: 1,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 72',
  },
  {
    name: 'SkinThickness',
    label: 'Triceps Skinfold Thickness',
    unit: 'mm',
    description: 'Triceps skinfold thickness measured with skinfold calipers to estimate subcutaneous fat.',
    clinical_reference: 'Typical adult range: 10–50 mm',
    min_value: 5,
    max_value: 110,
    step: 0.5,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 26',
  },
  {
    name: 'Insulin',
    label: '2-Hour Serum Insulin',
    unit: 'µU/mL',
    description: '2-hour post-load serum insulin level from a glucose tolerance blood panel.',
    clinical_reference: 'Typical 2-hr range: 16–166 µU/mL',
    min_value: 10,
    max_value: 900,
    step: 1,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 94',
  },
  {
    name: 'BMI',
    label: 'Body Mass Index (BMI)',
    unit: 'kg/m²',
    description: 'Weight in kilograms divided by height in meters squared (kg/m²).',
    clinical_reference: 'Normal: 18.5–24.9 | Overweight: 25.0–29.9 | Obese: ≥ 30.0',
    min_value: 10,
    max_value: 75,
    step: 0.1,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 26.4',
  },
  {
    name: 'DiabetesPedigreeFunction',
    label: 'Diabetes Pedigree Function',
    unit: 'score',
    description: 'Quantitative score reflecting genetic and family history predisposition to diabetes.',
    clinical_reference: 'Dataset range: 0.078–2.420 (higher = stronger family history)',
    min_value: 0.05,
    max_value: 3.0,
    step: 0.001,
    allow_zero: false,
    integer_only: false,
    placeholder: 'e.g., 0.351',
  },
  {
    name: 'Age',
    label: 'Age',
    unit: 'years',
    description: 'Patient age in completed years (training cohort consists of adults aged 21+).',
    clinical_reference: 'Training dataset cohort: 21–81 years',
    min_value: 18,
    max_value: 120,
    step: 1,
    allow_zero: false,
    integer_only: true,
    placeholder: 'e.g., 34',
  },
];

const INITIAL_VALUES = {
  Pregnancies: '',
  Glucose: '',
  BloodPressure: '',
  SkinThickness: '',
  Insulin: '',
  BMI: '',
  DiabetesPedigreeFunction: '',
  Age: '',
};

export default function PredictPage({ onPredictionComplete }) {
  const navigate = useNavigate();
  const [featureSpecs, setFeatureSpecs] = useState(FALLBACK_FEATURE_SPECS);
  const [modelName, setModelName] = useState('');
  const [sampleProfiles, setSampleProfiles] = useState([]);
  const [values, setValues] = useState(INITIAL_VALUES);
  const [fieldErrors, setFieldErrors] = useState({});
  const [submitError, setSubmitError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [activeSampleId, setActiveSampleId] = useState(null);

  useEffect(() => {
    let active = true;
    async function loadMeta() {
      try {
        const [infoRes, sampleRes] = await Promise.all([
          fetchModelInfo('diabetes'),
          fetchSampleInputs(),
        ]);
        if (!active) return;
        if (Array.isArray(infoRes?.features) && infoRes.features.length === 8) {
          setFeatureSpecs(infoRes.features);
        }
        if (infoRes?.selected_model_name) {
          setModelName(infoRes.selected_model_name);
        }
        if (Array.isArray(sampleRes?.profiles)) {
          setSampleProfiles(sampleRes.profiles);
        }
      } catch {
        // Fallback specs already cover schema cleanly; submission will report backend error if offline
      }
    }
    loadMeta();
    return () => {
      active = false;
    };
  }, []);

  function validateLocally(currentValues) {
    const errors = {};
    for (const spec of featureSpecs) {
      const raw = String(currentValues[spec.name] ?? '').trim();
      if (raw === '') {
        errors[spec.name] = `${spec.label} (${spec.unit}) is required — missing measurements are never fabricated.`;
        continue;
      }
      const num = Number(raw);
      if (!Number.isFinite(num)) {
        errors[spec.name] = `${spec.label} must be a valid number.`;
        continue;
      }
      if (!spec.allow_zero && num === 0) {
        errors[spec.name] = `${spec.label} cannot be 0 ${spec.unit}. Please enter an actual clinical reading.`;
        continue;
      }
      if (spec.integer_only && !Number.isInteger(num)) {
        errors[spec.name] = `${spec.label} must be a whole number.`;
        continue;
      }
      if (num < spec.min_value || num > spec.max_value) {
        errors[spec.name] = `Must be between ${spec.min_value} and ${spec.max_value} ${spec.unit}.`;
      }
    }
    return errors;
  }

  function handleChange(name, val) {
    setValues((prev) => ({ ...prev, [name]: val }));
    setActiveSampleId(null);
    if (fieldErrors[name]) {
      setFieldErrors((prev) => {
        const copy = { ...prev };
        delete copy[name];
        return copy;
      });
    }
    if (submitError) setSubmitError('');
  }

  function applySampleProfile(profile) {
    const next = {};
    for (const key of Object.keys(INITIAL_VALUES)) {
      next[key] = String(profile.inputs[key] ?? '');
    }
    setValues(next);
    setFieldErrors({});
    setSubmitError('');
    setActiveSampleId(profile.id);
  }

  function handleReset() {
    setValues(INITIAL_VALUES);
    setFieldErrors({});
    setSubmitError('');
    setActiveSampleId(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitError('');

    const localErrors = validateLocally(values);
    if (Object.keys(localErrors).length > 0) {
      setFieldErrors(localErrors);
      setSubmitError('Please fill in all 8 valid clinical measurements highlighted below.');
      return;
    }

    const payload = {};
    for (const spec of featureSpecs) {
      payload[spec.name] = Number(values[spec.name]);
    }

    setSubmitting(true);
    try {
      const result = await submitPrediction(payload);
      if (onPredictionComplete) {
        onPredictionComplete(result);
      }
      navigate('/results', { state: { result } });
    } catch (err) {
      if (err.fieldErrors) {
        setFieldErrors(err.fieldErrors);
      }
      setSubmitError(err.message || 'Prediction request failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  }

  const filledCount = featureSpecs.filter(
    (spec) => String(values[spec.name] ?? '').trim() !== ''
  ).length;

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700">
            <Stethoscope className="h-3.5 w-3.5" />
            Active Module: Type 2 Diabetes Risk Screening
          </div>
          <h1 className="mt-3 text-3xl font-extrabold tracking-tight text-navy-950 sm:text-4xl">
            Clinical Risk Assessment
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-600">
            Provide all 8 required clinical parameters below to execute live inference on our{' '}
            <strong className="text-navy-900">{modelName || 'trained scikit-learn'}</strong>{' '}
            machine learning pipeline.
          </p>
        </div>

        {/* Completion meter */}
        <div className="glass-card flex items-center gap-4 px-5 py-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Required Features
            </p>
            <p className="font-mono text-lg font-extrabold text-navy-900">
              {filledCount} / {featureSpecs.length}
            </p>
          </div>
          <div className="h-2.5 w-28 overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 transition-all duration-300"
              style={{ width: `${(filledCount / featureSpecs.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      <DisclaimerBanner compact />

      {/* Main Grid */}
      <div className="grid gap-8 lg:grid-cols-12">
        {/* Form Card */}
        <motion.form
          onSubmit={handleSubmit}
          noValidate
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6 sm:p-8 lg:col-span-8 space-y-6"
        >
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200/80 pb-4">
            <div>
              <h2 className="text-lg font-bold text-navy-900">Patient Clinical Measurements</h2>
              <p className="text-xs text-slate-500">
                All 8 measurements are mandatory. Zero is only valid for Number of Pregnancies.
              </p>
            </div>
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-100 hover:text-navy-900"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Clear Form
            </button>
          </div>

          <AnimatePresence>
            {submitError && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                role="alert"
                className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3.5 text-sm text-rose-900"
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
                <div>
                  <p className="font-bold">Validation or API Error</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-rose-800">{submitError}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* 8 Clinical Inputs */}
          <div className="grid gap-5 sm:grid-cols-2">
            {featureSpecs.map((spec) => {
              const error = fieldErrors[spec.name];
              const inputId = `field-${spec.name}`;
              const helpId = `help-${spec.name}`;
              const errorId = `error-${spec.name}`;

              return (
                <div
                  key={spec.name}
                  className={`rounded-2xl border p-4 transition-all ${
                    error
                      ? 'border-rose-300 bg-rose-50/40'
                      : 'border-slate-200/90 bg-slate-50/40 focus-within:border-blue-400 focus-within:bg-white focus-within:shadow-sm'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <label
                      htmlFor={inputId}
                      className="text-sm font-bold text-navy-900"
                    >
                      {spec.label}
                      <span className="ml-1 text-rose-500" aria-hidden="true">*</span>
                    </label>
                    <span className="shrink-0 rounded-md bg-navy-900/5 px-2 py-0.5 font-mono text-[11px] font-semibold text-navy-700">
                      {spec.unit}
                    </span>
                  </div>

                  <p id={helpId} className="mt-1 text-xs leading-relaxed text-slate-500">
                    {spec.description}
                  </p>

                  <div className="mt-3">
                    <input
                      id={inputId}
                      name={spec.name}
                      type="number"
                      inputMode="decimal"
                      min={spec.min_value}
                      max={spec.max_value}
                      step={spec.step}
                      required
                      value={values[spec.name]}
                      placeholder={spec.placeholder}
                      onChange={(e) => handleChange(spec.name, e.target.value)}
                      aria-invalid={Boolean(error)}
                      aria-describedby={error ? `${helpId} ${errorId}` : helpId}
                      className={`w-full rounded-xl border bg-white px-3.5 py-2.5 font-mono text-sm font-semibold text-navy-900 placeholder:font-sans placeholder:font-normal placeholder:text-slate-400 focus:outline-none focus:ring-2 ${
                        error
                          ? 'border-rose-400 focus:ring-rose-300'
                          : 'border-slate-300 focus:border-blue-500 focus:ring-cyan-200'
                      }`}
                    />
                  </div>

                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                    <span>{spec.clinical_reference}</span>
                  </div>

                  {error && (
                    <p
                      id={errorId}
                      role="alert"
                      className="mt-2 flex items-start gap-1.5 text-xs font-semibold text-rose-700"
                    >
                      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span>{error}</span>
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Submit Row */}
          <div className="flex flex-col items-stretch justify-between gap-4 border-t border-slate-200/80 pt-5 sm:flex-row sm:items-center">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Lock className="h-4 w-4 text-emerald-600" />
              <span>Processed strictly in-memory · Zero medical records persisted</span>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="btn-primary min-w-[230px] py-3.5 text-sm"
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running ML Inference…
                </>
              ) : (
                <>
                  <BrainCircuit className="h-4 w-4" />
                  Generate Risk Assessment
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </div>
        </motion.form>

        {/* Right Sidebar */}
        <aside className="space-y-6 lg:col-span-4">
          {/* Quick Demo Sample Profiles */}
          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center gap-2 text-blue-700">
              <Sparkles className="h-4 w-4" />
              <h2 className="text-sm font-extrabold uppercase tracking-wider">
                Sample Test Profiles
              </h2>
            </div>
            <p className="text-xs leading-relaxed text-slate-600">
              Click a clinical example below to pre-fill the 8 inputs and run a live prediction
              through the trained scikit-learn model.
            </p>

            <div className="space-y-2.5">
              {sampleProfiles.map((profile) => {
                const selected = activeSampleId === profile.id;
                return (
                  <button
                    key={profile.id}
                    type="button"
                    onClick={() => applySampleProfile(profile)}
                    className={`w-full rounded-xl border p-3.5 text-left transition-all ${
                      selected
                        ? 'border-cyan-400 bg-cyan-50/70 shadow-sm'
                        : 'border-slate-200 bg-white hover:border-blue-300 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-navy-900">{profile.title}</span>
                      {selected && <CheckCircle2 className="h-4 w-4 text-cyan-600" />}
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                      {profile.description}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1.5 font-mono text-[10px] text-slate-600">
                      <span className="rounded bg-slate-100 px-1.5 py-0.5">
                        Glu: {profile.inputs.Glucose}
                      </span>
                      <span className="rounded bg-slate-100 px-1.5 py-0.5">
                        BMI: {profile.inputs.BMI}
                      </span>
                      <span className="rounded bg-slate-100 px-1.5 py-0.5">
                        Age: {profile.inputs.Age}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Data Integrity Notice */}
          <div className="glass-card p-6 space-y-3">
            <div className="flex items-center gap-2 text-navy-900">
              <Info className="h-4 w-4 text-cyan-600" />
              <h3 className="text-sm font-bold">Why Are All 8 Values Required?</h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-600">
              To maintain clinical honesty, this system <strong>never silently substitutes</strong>{' '}
              missing patient measurements with synthetic defaults during live screening. While
              training-set median imputation handled historical missing zeros in the Pima dataset,
              individual risk assessment requires genuine measurements.
            </p>
          </div>

          {/* Extensible Multi-Disease Architecture Card */}
          <div className="glass-navy p-6 space-y-4">
            <div className="flex items-center gap-2 text-cyan-300">
              <FlaskConical className="h-4 w-4" />
              <h3 className="text-xs font-bold uppercase tracking-wider">
                Multi-Disease Architecture
              </h3>
            </div>
            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between rounded-xl border border-cyan-400/30 bg-cyan-500/10 px-3.5 py-2.5">
                <span className="font-semibold text-white">Diabetes Risk Screening</span>
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold text-emerald-300">
                  ACTIVE
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-slate-400">
                <span className="flex items-center gap-2">
                  <HeartPulse className="h-3.5 w-3.5" />
                  Cardiovascular Disease
                </span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold">
                  EXTENSIBLE
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-slate-400">
                <span className="flex items-center gap-2">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  Chronic Kidney Disease
                </span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold">
                  EXTENSIBLE
                </span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
