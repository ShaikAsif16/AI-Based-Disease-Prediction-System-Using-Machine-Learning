# Viva Voce Preparation — Questions & Answers

**Project:** AI-Based Disease Prediction System Using Machine Learning  
**Student:** Shaik Asif | **Hall Ticket:** 2403A52220 | **Subject:** AIDL

---

### Q1. Why did you choose the Pima Indians Diabetes Database, and what data quality issues did you discover during EDA?
**Answer:**  
The Pima Indians Diabetes Database (NIDDK / UCI) is a well-established benchmark clinical dataset containing 768 patient records and 8 physiological predictors. During exploratory data analysis, although Pandas reported zero explicit `NaN` values, we discovered physiologically impossible `0` values encoded as missing measurements across five clinical columns: `Insulin` (374 zeros, 48.7%), `SkinThickness` (227 zeros, 29.6%), `BloodPressure` (35 zeros, 4.6%), `BMI` (11 zeros, 1.4%), and `Glucose` (5 zeros, 0.7%). Meanwhile, `Pregnancies = 0` is medically valid and was preserved.

---

### Q2. How did you prevent data leakage during preprocessing and cross-validation?
**Answer:**  
We enforced two strict rules:
1. **Split before preprocessing:** The dataset is partitioned using `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` *before* computing any medians or scaling statistics.
2. **Encapsulation inside `sklearn.pipeline.Pipeline`:** Our custom `InvalidZeroToNaNTransformer`, `SimpleImputer(strategy="median")`, and `StandardScaler` are bundled inside a scikit-learn `Pipeline`. During 5-fold stratified cross-validation, each fold fits its own imputer and scaler solely on its training portion, so zero information from validation folds or the hold-out test set ever leaks into training.

---

### Q3. Why did you use median imputation instead of mean imputation or dropping rows?
**Answer:**  
Dropping all rows with invalid zeros would discard nearly half of the dataset (374 out of 768 rows have `Insulin = 0`), severely reducing sample size and statistical power. Median imputation was preferred over mean imputation because medical measurements like serum insulin (range 14–846 µU/mL) are strongly right-skewed; the median is robust to extreme physiological outliers.

---

### Q4. Support Vector Machine (SVM) achieved slightly higher test accuracy (74.68%) than Random Forest (74.03%). Why did you still select Random Forest?
**Answer:**  
In medical risk screening, **accuracy alone is misleading** when classes are imbalanced (65.1% negative vs. 34.9% positive). A False Negative (missing a diabetic patient) is much costlier clinically than a False Positive (flagging a patient for a confirmatory lab test). On the hold-out test set, SVM achieved only **57.41% Recall**, missing **23 out of 54** positive cases. **Random Forest** achieved **75.93% Recall** (catching 41 out of 54 positive cases — only 13 False Negatives), the highest **F1-Score (67.21%)**, and the highest **ROC-AUC (0.8276)**. Moreover, Random Forest was selected strictly using **5-fold cross-validation on the training set** (Clinical Screening Score = `0.45*Recall + 0.35*ROC_AUC + 0.20*F1 = 0.7745`) without peeking at the test set.

---

### Q5. Why did you use `stratify=y` during train/test splitting and `StratifiedKFold` during cross-validation?
**Answer:**  
Because the target classes are imbalanced (65.1% non-diabetic vs. 34.9% diabetic), an unstratified split could accidentally place disproportionately fewer positive cases in the test set or a CV fold, causing noisy and biased evaluation metrics. Stratification guarantees that the training split (614 rows, 214 positive), test split (154 rows, 54 positive), and every CV fold preserve the exact 34.9% class ratio.

---

### Q6. Why did you use `class_weight="balanced"` across your classifiers?
**Answer:**  
Standard classifiers minimize total misclassification rate, which naturally favors the majority negative class (65.1%). Setting `class_weight="balanced"` weights each class inversely proportional to its frequency, penalizing False Negatives more strongly and improving clinical sensitivity (Recall).

---

### Q7. During live website predictions, what happens if a user leaves a field empty or enters `0` for Glucose or BMI?
**Answer:**  
Both the React frontend and the Flask backend (`validate_diabetes_input` in `backend/ml/preprocessing.py`) **strictly reject** missing values or physiologically invalid zeros with HTTP `400` and clear field-specific validation errors. We intentionally do **not** silently substitute missing medical readings with dataset medians during individual risk screening, because fabricating a patient's missing glucose or BMI reading would produce a misleading risk score.

---

### Q8. How does the serialized model work in the Flask backend? Why save the full `Pipeline` with Joblib instead of just the classifier?
**Answer:**  
Saving the entire fitted `sklearn.pipeline.Pipeline` into `models/diabetes_pipeline.joblib` guarantees **training-serving skew elimination**. At inference time, raw user JSON inputs are validated into a DataFrame matching the exact 8-column schema and passed directly to `pipeline.predict()` and `pipeline.predict_proba()`. The exact same feature transformations (training-fitted scaler and column ordering) run identically in production as during evaluation.

---

### Q9. What does the ROC-AUC score of 0.8276 mean intuitively?
**Answer:**  
An ROC-AUC of **0.8276** means that if you randomly select one truly diabetic patient and one truly non-diabetic patient from the unseen hold-out test set, our Random Forest model assigns a higher estimated risk probability to the diabetic patient **82.76% of the time**, compared to 50.0% for random guessing.

---

### Q10. What are the top predictive features learned by the Random Forest model, and do they make medical sense?
**Answer:**  
Yes! The top Gini feature importances learned by our Random Forest are:
1. **Glucose (32.82%)** — 2-hour post-load plasma glucose directly reflects impaired glucose tolerance.
2. **BMI (17.72%)** — Adiposity strongly drives peripheral insulin resistance.
3. **Age (13.33%)** — Beta-cell function naturally declines with advancing age.
4. **Insulin (9.56%)** — Abnormal post-load insulin response signals insulin resistance or beta-cell dysfunction.

---

### Q11. How is your architecture designed to support additional diseases in the future?
**Answer:**  
In `backend/ml/predictor.py`, predictions are routed through a `DiseaseModelRegistry` keyed by `disease_id` (currently `"diabetes"`). Adding Cardiovascular Disease or Chronic Kidney Disease only requires training and saving a new scikit-learn pipeline artifact, defining its feature validation schema, registering it in `DiseaseModelRegistry`, and exposing the disease selector in the frontend—without rewriting the Flask API core or UI layout.

---

### Q12. What are the key limitations and ethical safeguards of this system?
**Answer:**  
- **Population bias:** The Pima dataset consists exclusively of female patients of Pima Indian heritage aged 21+, so generalization to other demographics requires external clinical validation.
- **Not a medical diagnosis:** All results use cautious risk wording ("Estimated Risk"), clearly explain limitations, recommend consulting a qualified physician for formal diagnostic tests (such as HbA1c or fasting plasma glucose), and display an explicit educational disclaimer.
- **Privacy by design:** Submitted health metrics are processed strictly in memory and are never saved to databases, files, or logs.
