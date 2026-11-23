import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  FileText,
  GraduationCap,
  Layers,
  Network,
  Server,
  Target,
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

const OBJECTIVES = [
  'Acquire and analyze a genuine clinical dataset (Pima Indians Diabetes Database) without synthetic data fabrication.',
  'Identify and properly handle physiologically invalid zero values in clinical features while preventing data leakage via scikit-learn Pipelines.',
  'Train, cross-validate, and benchmark four supervised learning algorithms: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).',
  'Select the best screening model using clinical validation metrics prioritizing Sensitivity (Recall) and ROC-AUC rather than accuracy alone.',
  'Serialize the complete preprocessing + classification pipeline with Joblib and expose validated RESTful endpoints using Python Flask.',
  'Build a modern, responsive, accessible React clinical dashboard with live inference, real-time analytics visualizations, and ethical medical disclaimers.',
  'Architect the backend registry modularly so additional disease models (Cardiovascular, Chronic Kidney Disease) can be integrated cleanly.',
];

const TECH_STACK = [
  {
    category: 'Frontend Interface',
    icon: Code2,
    items: ['React.js 18', 'Vite', 'Tailwind CSS', 'Framer Motion', 'Lucide Icons', 'Recharts', 'React Router'],
  },
  {
    category: 'Backend & API',
    icon: Server,
    items: ['Python 3', 'Flask REST API', 'Flask-CORS', 'Strict Input Validation', 'In-Memory Privacy Design', 'Pytest Automated Tests'],
  },
  {
    category: 'Machine Learning & EDA',
    icon: BrainCircuit,
    items: ['Scikit-learn Pipelines', 'Pandas & NumPy', 'Joblib Model Persistence', 'Stratified K-Fold CV', 'Matplotlib & Seaborn', 'Jupyter Notebook'],
  },
];

const ALGORITHMS = [
  {
    name: 'Logistic Regression',
    type: 'Linear Probabilistic Model',
    role: 'Strong interpretable baseline with L2 regularization and balanced class weights; quantifies linear log-odds contributions per feature.',
  },
  {
    name: 'Decision Tree (CART)',
    type: 'Non-Linear Rule Model',
    role: 'Depth-constrained tree (max_depth=5, min_samples_leaf=8) modeling explicit clinical threshold splits while mitigating overfitting.',
  },
  {
    name: 'Random Forest (Selected)',
    type: 'Bagged Tree Ensemble',
    role: '200 regularized trees capturing non-linear metabolic interactions (e.g., Glucose × BMI × Age); achieved the strongest CV clinical screening score, test Recall (75.9%), F1-Score (67.2%), and ROC-AUC (0.828).',
  },
  {
    name: 'Support Vector Machine (RBF)',
    type: 'Kernel Margin Classifier',
    role: 'Non-linear RBF kernel paired with cross-validated sigmoid probability calibration; demonstrates why accuracy alone can mask lower recall.',
  },
];

const ARCHITECTURE_LAYERS = [
  {
    step: '1',
    title: 'React Clinical Dashboard',
    detail: 'Validated 8-feature form, responsive views, Recharts analytics, Framer Motion UX',
  },
  {
    step: '2',
    title: 'Flask REST API Layer',
    detail: 'GET /api/health, /api/model-info, /api/model-metrics & POST /api/predict (CORS + JSON validation)',
  },
  {
    step: '3',
    title: 'Disease Model Registry',
    detail: 'Modular multi-disease service loading saved Joblib pipelines purely in memory (zero DB storage)',
  },
  {
    step: '4',
    title: 'Trained Scikit-Learn Pipeline',
    detail: 'InvalidZeroToNaNTransformer → Median Imputer → StandardScaler → Selected Random Forest Classifier',
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-7xl space-y-10 px-4 py-10 sm:px-6 lg:px-8">
      {/* Academic Header Banner */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-navy relative overflow-hidden p-7 sm:p-10"
      >
        <div className="pointer-events-none absolute -right-12 -top-12 h-64 w-64 rounded-full bg-cyan-500/20 blur-3xl" />
        <div className="relative grid gap-8 lg:grid-cols-12 lg:items-center">
          <div className="space-y-3 lg:col-span-8">
            <div className="inline-flex items-center gap-2 rounded-full bg-cyan-500/20 px-3.5 py-1 text-xs font-bold text-cyan-300 ring-1 ring-cyan-400/30">
              <GraduationCap className="h-4 w-4" />
              Artificial Intelligence &amp; Deep Learning (AIDL)
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              AI-Based Disease Prediction System Using Machine Learning
            </h1>
            <p className="text-sm leading-relaxed text-slate-300">
              A complete, reproducible full-stack machine learning web application combining
              authentic clinical data science with a premium healthcare screening interface.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 lg:col-span-4 space-y-2.5 text-sm">
            <div className="flex justify-between border-b border-white/10 pb-2">
              <span className="text-slate-400">Student Name</span>
              <span className="font-bold text-white">Shaik Asif</span>
            </div>
            <div className="flex justify-between border-b border-white/10 pb-2">
              <span className="text-slate-400">Hall Ticket Number</span>
              <span className="font-mono font-bold text-cyan-300">2403A52220</span>
            </div>
            <div className="flex justify-between border-b border-white/10 pb-2">
              <span className="text-slate-400">Subject</span>
              <span className="font-semibold text-white">AIDL</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Project Type</span>
              <span className="font-semibold text-emerald-300">Full-Stack ML Web App</span>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Problem Statement & Project Objectives */}
      <div className="grid gap-8 lg:grid-cols-12">
        <section className="glass-card p-6 sm:p-7 lg:col-span-5 space-y-4">
          <div className="flex items-center gap-2.5 text-blue-700">
            <FileText className="h-5 w-5" />
            <h2 className="text-xl font-bold text-navy-950">Problem Statement</h2>
          </div>
          <p className="text-sm leading-relaxed text-slate-600">
            Chronic metabolic conditions such as Type 2 Diabetes often progress silently for years
            before formal diagnosis, leading to preventable cardiovascular, renal, and neurological
            complications. Traditional manual risk screening can miss subtle non-linear interactions
            among biomarkers such as post-load glucose, BMI, age, serum insulin, and hereditary
            predisposition.
          </p>
          <p className="text-sm leading-relaxed text-slate-600">
            At the same time, many academic demo websites either rely on hardcoded rules, leak test
            data during preprocessing, or optimize accuracy alone—ignoring the high clinical cost
            of missed positive cases (False Negatives). This project addresses both gaps by building
            a genuinely trained, leakage-free, sensitivity-prioritized ML screening web application.
          </p>
        </section>

        <section className="glass-card p-6 sm:p-7 lg:col-span-7 space-y-4">
          <div className="flex items-center gap-2.5 text-cyan-700">
            <Target className="h-5 w-5" />
            <h2 className="text-xl font-bold text-navy-950">Project Objectives</h2>
          </div>
          <ul className="space-y-3 text-sm text-slate-600">
            {OBJECTIVES.map((obj, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                <span className="leading-relaxed">{obj}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      {/* Technologies Used */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-blue-600" />
          <h2 className="text-2xl font-extrabold text-navy-950">Technologies Used</h2>
        </div>
        <div className="grid gap-6 md:grid-cols-3">
          {TECH_STACK.map(({ category, icon: Icon, items }) => (
            <div key={category} className="glass-card p-6 space-y-4">
              <div className="flex items-center gap-2.5">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="text-base font-bold text-navy-900">{category}</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {items.map((tech) => (
                  <span
                    key={tech}
                    className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-navy-800"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Dataset Description & ML Algorithms */}
      <div className="grid gap-8 lg:grid-cols-12">
        <section className="glass-card p-6 sm:p-7 lg:col-span-5 space-y-4">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-cyan-600" />
            <h2 className="text-xl font-bold text-navy-950">Dataset Description</h2>
          </div>
          <p className="text-sm leading-relaxed text-slate-600">
            The model is trained on the authentic{' '}
            <strong className="text-navy-900">Pima Indians Diabetes Database</strong> originally
            collected by the National Institute of Diabetes and Digestive and Kidney Diseases
            (NIDDK) and published via UCI / Kaggle.
          </p>
          <dl className="divide-y divide-slate-200/80 rounded-xl border border-slate-200 bg-slate-50/60 text-xs">
            <div className="flex justify-between px-4 py-2.5">
              <dt className="text-slate-500">Total Records</dt>
              <dd className="font-mono font-bold text-navy-900">768 Female Patients (Age 21+)</dd>
            </div>
            <div className="flex justify-between px-4 py-2.5">
              <dt className="text-slate-500">Input Features</dt>
              <dd className="font-mono font-bold text-navy-900">8 Clinical Attributes</dd>
            </div>
            <div className="flex justify-between px-4 py-2.5">
              <dt className="text-slate-500">Class Balance</dt>
              <dd className="font-mono font-bold text-navy-900">500 Neg (65.1%) / 268 Pos (34.9%)</dd>
            </div>
            <div className="flex justify-between px-4 py-2.5">
              <dt className="text-slate-500">Hidden Missing Zeros</dt>
              <dd className="font-mono font-bold text-blue-700">
                Insulin (374), Skin (227), BP (35), BMI (11), Glu (5)
              </dd>
            </div>
            <div className="flex justify-between px-4 py-2.5">
              <dt className="text-slate-500">Split Strategy</dt>
              <dd className="font-mono font-bold text-navy-900">Stratified 80% Train / 20% Test</dd>
            </div>
          </dl>
        </section>

        <section className="glass-card p-6 sm:p-7 lg:col-span-7 space-y-4">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-blue-600" />
            <h2 className="text-xl font-bold text-navy-950">Machine Learning Algorithms Evaluated</h2>
          </div>
          <div className="grid gap-3.5 sm:grid-cols-2">
            {ALGORITHMS.map((alg) => (
              <div
                key={alg.name}
                className={`rounded-2xl border p-4 space-y-1.5 ${
                  alg.name.includes('Selected')
                    ? 'border-cyan-300 bg-cyan-50/50'
                    : 'border-slate-200 bg-slate-50/60'
                }`}
              >
                <p className="text-[11px] font-bold uppercase tracking-wider text-blue-700">
                  {alg.type}
                </p>
                <h3 className="text-sm font-extrabold text-navy-900">{alg.name}</h3>
                <p className="text-xs leading-relaxed text-slate-600">{alg.role}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* System Architecture */}
      <section className="glass-card p-6 sm:p-8 space-y-6">
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-blue-600" />
          <div>
            <h2 className="text-xl font-bold text-navy-950">System Architecture</h2>
            <p className="text-xs text-slate-500">
              End-to-end separation of presentation, API validation, model registry, and ML artifact persistence
            </p>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          {ARCHITECTURE_LAYERS.map((layer) => (
            <div
              key={layer.step}
              className="relative rounded-2xl border border-slate-200 bg-gradient-to-b from-white to-slate-50 p-5 space-y-2"
            >
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-navy-900 font-mono text-xs font-bold text-cyan-300">
                {layer.step}
              </span>
              <h3 className="text-sm font-extrabold text-navy-900">{layer.title}</h3>
              <p className="text-xs leading-relaxed text-slate-600">{layer.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Project Conclusion */}
      <section className="glass-card p-6 sm:p-8 space-y-4">
        <h2 className="text-xl font-bold text-navy-950">Project Conclusion &amp; Future Scope</h2>
        <p className="text-sm leading-relaxed text-slate-600">
          This project demonstrates how responsible machine learning engineering can power a clean,
          transparent healthcare screening application. By strictly separating training and hold-out
          test sets prior to preprocessing, handling physiologically invalid zero measurements
          inside a scikit-learn Pipeline, and selecting the final Random Forest classifier on
          sensitivity-oriented cross-validation metrics, the system achieves reliable risk
          stratification without data leakage or inflated claims.
        </p>
        <p className="text-sm leading-relaxed text-slate-600">
          Because inference is managed through a modular <code className="font-mono text-xs">DiseaseModelRegistry</code>{' '}
          and schema-driven API, future iterations can seamlessly incorporate Cardiovascular Disease
          and Chronic Kidney Disease screening models, explainable AI (SHAP feature attributions),
          and probability calibration monitoring.
        </p>

        <div className="flex flex-wrap gap-3 pt-2">
          <Link to="/predict" className="btn-primary">
            Try Live Diabetes Prediction
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link to="/analytics" className="btn-secondary">
            Inspect Model Evaluation Metrics
          </Link>
        </div>
      </section>

      <DisclaimerBanner compact />
    </div>
  );
}
