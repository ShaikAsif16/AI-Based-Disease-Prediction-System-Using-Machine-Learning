# Academic Project Report: AI-Based Disease Prediction System Using Machine Learning

## Student & Course Information
- **Student Name:** Shaik Asif
- **Hall Ticket Number:** 2403A52220
- **Subject:** Artificial Intelligence & Deep Learning (AIDL)
- **Project Title:** AI-Based Disease Prediction System Using Machine Learning
- **Project Category:** Full-Stack Machine Learning Web Application

---

## 1. Abstract

Early risk screening for chronic metabolic diseases such as Type 2 Diabetes is critical to preventing long-term cardiovascular, renal, and neurological complications. This project presents an end-to-end **AI-Based Disease Prediction System** powered by a genuinely trained scikit-learn machine learning pipeline and delivered through a modern React.js and Python Flask web application.

Using the authentic **Pima Indians Diabetes Database** ($N = 768$ clinical records), the pipeline systematically identifies physiologically invalid zero values in five clinical measurements, enforces a strict stratified train/test partition prior to any imputation or scaling to prevent data leakage, and benchmarks four supervised learning algorithms: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**. Recognizing that missed positive cases (False Negatives) carry high clinical cost in health screening, model selection is governed by a 5-fold stratified cross-validation screening criterion prioritizing **Recall (Sensitivity)**, **ROC-AUC**, and **F1-Score** rather than accuracy alone. The selected **Random Forest** pipeline achieves **75.93% hold-out test Recall**, **0.8276 ROC-AUC**, **67.21% F1-Score**, and **74.03% Accuracy**, and is served via a privacy-preserving, in-memory REST API integrated with an interactive clinical dashboard.

---

## 2. Problem Statement

Chronic non-communicable diseases often progress without obvious symptoms during early stages. While clinical biomarkers—such as 2-hour post-load plasma glucose, body mass index (BMI), diastolic blood pressure, serum insulin, age, and family history—carry strong predictive signal, their interactions are non-linear and difficult to assess manually. Furthermore, common pitfalls in student ML health projects include:

1. **Data leakage** caused by imputing missing values or scaling features across the full dataset prior to splitting into training and testing sets.
2. **Hidden missing values**, where clinical zeros (e.g., `Glucose = 0` or `BMI = 0`) are mistakenly treated as real numbers.
3. **Accuracy bias**, where models are chosen solely by overall accuracy despite low sensitivity on imbalanced medical datasets.
4. **Silent imputation during inference**, where web forms silently fill empty medical fields with averages instead of requiring authentic clinical measurements.

This project solves all four challenges through disciplined machine learning engineering and transparent full-stack design.

---

## 3. Dataset Description & Exploratory Data Analysis

- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / UCI Machine Learning Repository (`pima_indians_diabetes.csv`).
- **Records:** 768 female patients of Pima Indian heritage, aged 21 years and older.
- **Target Variable (`Outcome`):**
  - `0` (Non-Diabetic): **500 samples (65.10%)**
  - `1` (Diabetic): **268 samples (34.90%)**
- **Explicit `NaN` values:** 0 across all 8 columns.
- **Physiologically Invalid Zero Values Detected:**
  - `Insulin`: **374** (48.70%)
  - `SkinThickness`: **227** (29.56%)
  - `BloodPressure`: **35** (4.56%)
  - `BMI`: **11** (1.43%)
  - `Glucose`: **5** (0.65%)
  - (`Pregnancies = 0` is physiologically valid and left unchanged.)

- **Top Correlations with Diabetes Outcome (after masking invalid zeros as `NaN`):**
  - `Glucose`: **+0.4947**
  - `BMI`: **+0.3137**
  - `Insulin`: **+0.3035**
  - `SkinThickness`: **+0.2595**
  - `Age`: **+0.2384**

---

## 4. Preprocessing & Data Leakage Prevention Methodology

1. **Stratified Train/Test Split (`random_state = 42`):**
   - Training set: **614 samples** (400 Negative, 214 Positive — 34.85% positive)
   - Hold-out test set: **154 samples** (100 Negative, 54 Positive — 35.06% positive)
2. **Encapsulated Scikit-Learn Pipeline:**
   Every model is wrapped inside a unified `sklearn.pipeline.Pipeline`:
   - **Step 1 (`InvalidZeroToNaNTransformer`):** Custom scikit-learn transformer replacing `0` with `NaN` strictly in `["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]`.
   - **Step 2 (`SimpleImputer(strategy="median")`):** Computes feature medians strictly on the training split (or training fold during cross-validation). Median imputation is robust to skewed medical distributions such as serum insulin.
   - **Step 3 (`StandardScaler`):** Standardizes features to zero mean and unit variance using training-only parameters.
   - **Step 4 (`Classifier`):** Balanced class weighting (`class_weight="balanced"`) to counteract class imbalance and improve clinical sensitivity.
3. **Training-Only Fitted Medians:**
   `Pregnancies: 3.0`, `Glucose: 117.0`, `BloodPressure: 72.0`, `SkinThickness: 29.0`, `Insulin: 125.0`, `BMI: 32.3`, `DiabetesPedigreeFunction: 0.3825`, `Age: 29.0`.

---

## 5. Model Training, Validation & Evaluation Results

All four models were evaluated using **5-fold Stratified Cross-Validation** on the 614-sample training set for unbiased model selection, followed by final evaluation on the untouched 154-sample hold-out test set.

### Clinical Selection Criterion (Training-Set 5-Fold CV Only)
$$\text{Clinical Screening Score} = 0.45 \times \text{CV\_Recall} + 0.35 \times \text{CV\_ROC\_AUC} + 0.20 \times \text{CV\_F1}$$

### Comparative Performance Table

| Algorithm | 5-Fold CV Score | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test Specificity | Test F1-Score | Test ROC-AUC | Test False Negatives |
|---|---|---|---|---|---|---|---|---|
| **Random Forest (Selected)** | **0.7745** | 74.03% | 60.29% | **75.93%** | 73.00% | **67.21%** | **0.8276** | **13 / 54** |
| Logistic Regression | 0.7455 | 73.38% | 60.32% | 70.37% | 75.00% | 64.96% | 0.8126 | 16 / 54 |
| Decision Tree (max_depth=5) | 0.7328 | 72.73% | 60.00% | 66.67% | 76.00% | 63.16% | 0.7935 | 18 / 54 |
| Support Vector Machine (RBF) | 0.6979 | **74.68%** | **65.96%** | 57.41% | **84.00%** | 61.39% | 0.8139 | 23 / 54 |

### Why Random Forest Was Selected Over SVM Despite SVM's Slightly Higher Accuracy
Although SVM achieved marginally higher hold-out accuracy (74.68% vs. 74.03%), it achieved only **57.41% Recall**, missing **23 out of 54** diabetic patients in the test set. In contrast, **Random Forest** achieved the highest 5-fold CV Clinical Screening Score (**0.7745**), the highest hold-out Recall (**75.93%**, detecting **41 out of 54** positive cases and cutting missed cases by 43% compared to SVM), the highest F1-Score (**67.21%**), and the highest ROC-AUC (**0.8276**).

### Random Forest Hold-Out Confusion Matrix (N = 154)
- **True Negatives (TN):** 73
- **False Positives (FP):** 27
- **False Negatives (FN):** 13
- **True Positives (TP):** 41

### Top Learned Feature Importances (Random Forest Gini Importance)
1. **Glucose:** 32.82%
2. **BMI:** 17.72%
3. **Age:** 13.33%
4. **Insulin:** 9.56%

---

## 6. System Architecture

1. **Presentation Layer (`frontend/`):** React.js + Tailwind CSS + Framer Motion + Recharts + Lucide icons. Five full views: Home Dashboard, Disease Prediction Form, Prediction Results View, Model Analytics Dashboard, and About Project Page.
2. **API Layer (`backend/routes/api.py`, `backend/app.py`):** Python Flask + Flask-CORS serving `/api/health`, `/api/model-info`, `/api/model-metrics`, `/api/sample-inputs`, `/api/predict`, and `/api/figures/<name>`.
3. **Inference & Multi-Disease Registry (`backend/ml/predictor.py`):** `DiseaseModelRegistry` loads `models/diabetes_pipeline.joblib` purely in memory, enforces strict input validation (rejecting missing fields or invalid zeros), returns predicted class + probability + cautious risk band + clinical reference context + medical disclaimer, and stores zero user medical data.
4. **ML Training & Reporting (`backend/ml/train.py`, `reports/figures/`):** Reproducible training script generating serialized artifacts (`diabetes_pipeline.joblib`, `model_metadata.json`, `evaluation_metrics.json`) and 6 Matplotlib/Seaborn analytical charts.

---

## 7. Ethical Considerations & Limitations

- **Educational Screening Only:** The application explicitly communicates that it is an academic demonstration and **not a medical diagnostic tool**.
- **Cohort Generalizability:** Because the Pima dataset represents female patients of Pima Indian heritage aged 21+, real-world clinical deployment would require external validation across diverse multi-ethnic, mixed-sex populations.
- **Data Privacy:** No user-entered clinical metrics are written to logs, files, or databases.

---

## 8. Conclusion & Future Enhancements

This project successfully delivers a complete, reproducible, and ethically designed AI-Based Disease Prediction System. Future extensions enabled by the modular `DiseaseModelRegistry` architecture include adding Cardiovascular Disease and Chronic Kidney Disease screening models, per-patient SHAP explainability charts, and probability calibration monitoring.
