# Dataset Acquisition Instructions

## Dataset Overview

- **Dataset Name:** Pima Indians Diabetes Database
- **Original Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Public Repositories:** UCI Machine Learning Repository / Kaggle (`uciml/pima-indians-diabetes-database`) / OpenML (`data_id=37`)
- **Samples:** 768 female patients of Pima Indian heritage, aged 21 years and older
- **Features:** 8 clinical predictor variables + 1 binary outcome (`Outcome`: `0` = Non-diabetic, `1` = Diabetic)
- **Class Distribution:** 500 Negative (`65.1%`) and 268 Positive (`34.9%`)

## Features

| Column | Description | Unit | Can Be 0 Physiologically? |
|---|---|---|---|
| `Pregnancies` | Number of times pregnant | Count | Yes |
| `Glucose` | Plasma glucose concentration at 2 hours in an oral glucose tolerance test (OGTT) | mg/dL | **No** (0 represents missing) |
| `BloodPressure` | Diastolic blood pressure | mm Hg | **No** (0 represents missing) |
| `SkinThickness` | Triceps skinfold thickness | mm | **No** (0 represents missing) |
| `Insulin` | 2-hour serum insulin | µU/mL | **No** (0 represents missing) |
| `BMI` | Body mass index ($\text{weight in kg} / (\text{height in m})^2$) | kg/m² | **No** (0 represents missing) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (hereditary risk score) | Score | **No** |
| `Age` | Age of the patient | Years | **No** |
| `Outcome` | Class variable (`0` = Negative, `1` = Positive) | Binary | Target variable |

## Option 1: Automatic Download (Recommended)

From the project root (`ai-disease-prediction/`), run:

```bash
python dataset/download_dataset.py
```

The script downloads the canonical Pima Indians Diabetes dataset from verified public UCI/OpenML mirrors, validates the exact schema (768 rows, 9 columns, 500 negative / 268 positive), and saves it to:

```
dataset/pima_indians_diabetes.csv
```

## Option 2: Manual Download from Kaggle

1. Visit: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Sign in to Kaggle and download `diabetes.csv`.
3. Rename the downloaded file to `pima_indians_diabetes.csv` and place it in:
   ```
   ai-disease-prediction/dataset/pima_indians_diabetes.csv
   ```
4. Verify the file by running:
   ```bash
   python dataset/download_dataset.py
   ```

## Option 3: Manual Download from OpenML

1. Visit: https://www.openml.org/search?type=data&id=37
2. Download the dataset CSV and ensure columns are named:
   `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`
   with `Outcome` encoded as `0` / `1`.
3. Save to `dataset/pima_indians_diabetes.csv`.

## Data Integrity Policy

This project **never** substitutes synthetic or randomly generated data if the real dataset is missing. The training script verifies the presence and schema of `dataset/pima_indians_diabetes.csv` before proceeding.
