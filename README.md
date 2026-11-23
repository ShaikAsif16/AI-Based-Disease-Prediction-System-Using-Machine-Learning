# AI-Based Disease Prediction System Using Machine Learning

> **Student Name:** Shaik Asif  
> **Hall Ticket Number:** 2403A52220  
> **Subject:** Artificial Intelligence & Deep Learning (AIDL)  
> **Project Type:** Full-Stack Machine Learning Web Application

A complete, reproducible full-stack healthcare risk screening web application powered by a **genuinely trained scikit-learn machine learning pipeline** on the **Pima Indians Diabetes Database** (NIDDK / UCI), integrated with a Python Flask REST API and a modern React + Tailwind CSS + Framer Motion + Recharts clinical dashboard.

> **Medical Disclaimer:** This application is an **educational screening demonstration** built for academic evaluation and **not a medical diagnosis**. It does not store user medical information. Always consult a qualified healthcare professional for medical advice or diagnostic testing.

---

## Key Highlights

- **100% Authentic ML Pipeline:** Trained on the real Pima Indians Diabetes dataset (`768 rows × 9 columns`). Zero hardcoded predictions, zero random outputs, zero synthetic data substitution.
- **Zero Data Leakage:** Stratified 80/20 split (`random_state=42`) performed *before* replacing physiologically invalid zeros (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`), fitting training-only median imputation, and applying `StandardScaler` inside a unified scikit-learn `Pipeline`.
- **Sensitivity-First Model Selection:** Evaluates **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** using 5-fold Stratified Cross-Validation on the training set, selecting **Random Forest** (`CV Score = 0.7745`, `Hold-Out Test Recall = 75.93%`, `ROC-AUC = 0.8276`, `F1 = 67.21%`, `Accuracy = 74.03%`).
- **Strict Input Validation & Privacy:** Rejects missing measurements and physiologically invalid zeros with field-level feedback (never silently fabricating inputs). Submitted health inputs are processed strictly in memory and never stored.
- **Extensible Multi-Disease Architecture:** Backend uses a `DiseaseModelRegistry` supporting Diabetes today and ready for Cardiovascular and Chronic Kidney Disease models.

---

## Project Structure

```text
ai-disease-prediction/
├── backend/
│   ├── ml/
│   │   ├── preprocessing.py      # Feature schema, InvalidZeroToNaNTransformer, strict validation
│   │   ├── train.py              # Reproducible EDA, 5-fold CV, 4-model training, Joblib export & plots
│   │   └── predictor.py          # Multi-disease registry & real-time inference engine
│   ├── routes/
│   │   └── api.py                # REST API endpoints (/health, /predict, /model-info, /model-metrics)
│   ├── tests/
│   │   ├── test_ml_pipeline.py   # Preprocessing & model artifact automated tests
│   │   └── test_api.py           # Flask endpoint integration tests
│   └── app.py                    # Flask app factory & CORS entrypoint
├── dataset/
│   ├── download_dataset.py       # Fetches & verifies authentic Pima dataset (never generates fake data)
│   ├── DATASET_INSTRUCTIONS.md   # Manual Kaggle / UCI / OpenML acquisition guide
│   └── pima_indians_diabetes.csv # Authentic NIDDK/UCI dataset (768 rows)
├── frontend/
│   ├── src/
│   │   ├── components/           # Navbar, Footer, DisclaimerBanner, ProbabilityGauge, ConfusionMatrixCard
│   │   ├── pages/                # HomePage, PredictPage, ResultsPage, AnalyticsPage, AboutPage
│   │   ├── services/api.js       # Typed API client connecting to Flask backend
│   │   └── App.jsx               # Router & live API health integration
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── diabetes_pipeline.joblib  # Serialized fitted scikit-learn Pipeline
│   ├── model_metadata.json       # Feature specs, selection rationale, training medians
│   └── evaluation_metrics.json   # Real CV & hold-out metrics, confusion matrices, ROC curves
├── notebooks/
│   └── 01_eda_and_model_training.ipynb
├── reports/
│   ├── figures/                  # 6 Matplotlib/Seaborn evaluation plots (PNG)
│   ├── ACADEMIC_PROJECT_REPORT.md
│   ├── VIVA_QA.md
│   └── SAMPLE_TEST_INPUTS.json
├── requirements.txt
└── README.md
```

---

## Model Evaluation Summary (Hold-Out Test Set, N = 154)

| Algorithm | 5-Fold CV Clinical Score | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC | Missed Cases (FN) |
|---|---|---|---|---|---|---|---|
| **Random Forest (Selected)** | **0.7745** | 74.03% | 60.29% | **75.93%** | **67.21%** | **0.8276** | **13 / 54** |
| Logistic Regression | 0.7455 | 73.38% | 60.32% | 70.37% | 64.96% | 0.8126 | 16 / 54 |
| Decision Tree | 0.7328 | 72.73% | 60.00% | 66.67% | 63.16% | 0.7935 | 18 / 54 |
| Support Vector Machine (RBF) | 0.6979 | 74.68% | 65.96% | 57.41% | 61.39% | 0.8139 | 23 / 54 |

---

## Local Installation & Running Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Node.js 18+ & npm (tested on Node 22)

### 1. Set Up Python Environment & Install Dependencies
```bash
cd ai-disease-prediction
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Acquire / Verify the Authentic Dataset
```bash
python dataset/download_dataset.py
```
If offline, see `dataset/DATASET_INSTRUCTIONS.md` for manual download steps from Kaggle or OpenML.

### 3. Train, Evaluate & Save the ML Pipeline
```bash
python backend/ml/train.py
```
This reproduces the stratified split, 5-fold cross-validation, hold-out evaluation, saves `models/diabetes_pipeline.joblib`, `models/model_metadata.json`, `models/evaluation_metrics.json`, and generates 6 PNG evaluation figures in `reports/figures/`.

### 4. Run Automated Tests
```bash
pytest backend/tests -v
```

### 5. Launch the Flask Backend API (Terminal 1)
```bash
python backend/app.py
```
The API runs at `http://127.0.0.1:5000`.

### 6. Launch the React Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173` in your browser! (Vite proxies `/api/*` requests directly to `http://127.0.0.1:5000`.)

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, active model name, privacy policy |
| `GET` | `/api/model-info` | Selected model metadata, feature schema, training config |
| `GET` | `/api/model-metrics` | Genuine CV & test metrics, confusion matrices, ROC curves, EDA stats |
| `GET` | `/api/sample-inputs` | Demo clinical profiles for testing live ML inference |
| `POST` | `/api/predict` | Validates all 8 clinical inputs and returns real ML risk prediction |
| `GET` | `/api/figures/<name>` | Serves generated Matplotlib/Seaborn training plots |

### Example `POST /api/predict` Request
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 176,
    "BloodPressure": 86,
    "SkinThickness": 36,
    "Insulin": 215,
    "BMI": 35.4,
    "DiabetesPedigreeFunction": 0.782,
    "Age": 52
  }'
```

---

## Academic Documentation
- [Academic Project Report](reports/ACADEMIC_PROJECT_REPORT.md)
- [Viva Voce Preparation Q&A](reports/VIVA_QA.md)
- [Dataset Acquisition Instructions](dataset/DATASET_INSTRUCTIONS.md)
- [Sample Test Payloads](reports/SAMPLE_TEST_INPUTS.json)
- [EDA & Training Notebook](notebooks/01_eda_and_model_training.ipynb)
