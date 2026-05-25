# =========================
# QA AND TEST LOG
# =========================

This document records all testing, debugging, and validation activities performed on the data science pipeline. It covers data integrity checks, feature engineering validation, model training verification, evaluation metric analysis, and full end-to-end pipeline execution results. Each section includes quantitative evidence from actual test runs.


# =========================
# PROJECT OVERVIEW
# =========================

The project is a regression-based machine learning pipeline that predicts student academic performance using three separate datasets: Gaming, Mental Health, and Student Performance. The pipeline follows these stages:

    1. Data Loading and Cleaning (load_and_clean.py)
    2. Feature Engineering (features.py)
    3. Model Training with Cross-Validation and GridSearchCV (models.py, main.py)
    4. Evaluation with RMSE, MAE, R2, Adjusted R2 (evaluation.py)
    5. Explainability with Feature Importance, Permutation Importance, and SHAP (explainability.py)

Six regression models are trained and compared for each dataset:
Linear Regression, Decision Tree, Random Forest, SVR, KNN, and Gradient Boosting.


# =========================
# TASK CHECKLIST
# =========================

- [x] Verified that `load_and_clean.py` executes without errors across all three datasets.
- [x] Confirmed that missing values are properly handled (dropped rows with NaN).
- [x] Ensured that cleaned CSV outputs have correct column names, data types, and structure.
- [x] Verified that `features.py` generates new engineered features without NaN or Infinity values.
- [x] Confirmed that feature statistics (mean, std, min, max) fall within logical ranges.
- [x] Checked that no data leakage occurs from target variables into the feature set.
- [x] Ensured that `student_id`, `source`, and `cgpa_range` columns are dropped before model training.
- [x] Verified that One-Hot Encoding is applied correctly to categorical columns.
- [x] Confirmed that StandardScaler is used inside the pipeline (not applied before train/test split).
- [x] Verified that `models.py` defines six regression pipelines with GridSearchCV hyperparameter tuning.
- [x] Confirmed that 5-fold cross-validation is used consistently across all models.
- [x] Ensured that `train.py` (via main.py) executes model training without throwing exceptions.
- [x] Verified that `evaluation.py` computes RMSE, MAE, R2, and Adjusted R2 correctly.
- [x] Confirmed that evaluation metrics are saved as CSV files and comparison plots are generated.
- [x] Verified that Actual vs Predicted scatter plots are generated for all models.
- [x] Checked that Feature Importance, Permutation Importance, and SHAP plots are saved to the qa_debug_and_test_outputs/reports/ directory.
- [x] Confirmed that the end-to-end pipeline runs sequentially without errors.
- [x] Verified that `requirements.txt` includes all necessary dependencies.
- [x] Confirmed that all output files (CSV, PNG) are saved to the correct directories and relocated to qa_debug_and_test_outputs/.


# =========================
# STAGE 1: DATA LOADING AND CLEANING VALIDATION
# =========================

## Raw Data Inspection

Three raw CSV files were loaded from `data/raw/`:

    Gaming Dataset:
        - File: gaming.csv (614,879 bytes)
        - Shape: 8000 rows x 14 columns
        - Missing Values: 0
        - Columns: student_id, age, gender, gaming_hours, study_hours,
          sleep_hours, attendance, gaming_genre, social_activity,
          device_usage, reaction_time_ms, addiction_score, stress_level, grades
        - Target Variable: grades (float64)

    Mental Health Dataset:
        - File: mental.csv (7,339 bytes)
        - Shape: 101 rows x 11 columns
        - Missing Values: 1 (one row dropped during cleaning)
        - Columns: Timestamp, Choose your gender, Age, What is your course?,
          Your current year of Study, What is your CGPA?, Marital status,
          Do you have Depression?, Do you have Anxiety?,
          Do you have Panic attack?, Did you seek any specialist for a treatment?
        - Target Variable: final_cgpa (derived from CGPA range midpoint)

    Student Performance Dataset:
        - File: student.csv (268,149 bytes)
        - Shape: 5000 rows x 10 columns
        - Missing Values: 0
        - Columns: Student_ID, Gender, Age, Major, Attendance_Pct,
          Study_Hours_Per_Day, Previous_CGPA, Sleep_Hours,
          Social_Hours_Week, Final_CGPA
        - Target Variable: Final_CGPA (float64)

## Cleaning Verification Results

After running `load_and_clean.py`, the following cleaned datasets were produced in `data/processed/`:

    Cleaned Gaming:
        - Shape: 8000 rows x 20 columns (6 new columns added)
        - Missing Values: 0
        - Changes Applied:
            * student_id prefixed with "gaming_" for uniqueness
            * Column renames: study_hours -> study_hours_per_day,
              sleep_hours -> sleep_hours_per_day, social_activity -> social_hours_day
            * Gender mapped to numeric (Female=0, Male=1, Other=2)
            * One-Hot encoded: is_female, is_male, is_other
            * Derived: social_hours_week = social_hours_day * 7
            * Derived: final_cgpa = grades / 25
            * Source column added: "gaming"

    Cleaned Mental:
        - Shape: 100 rows x 16 columns (1 row dropped due to missing value)
        - Missing Values: 0
        - Changes Applied:
            * Timestamp column dropped (irrelevant for analysis)
            * All columns renamed to standardized short names
            * Yes/No columns converted to 1/0 (marital_status, depression,
              anxiety, panic_attack, seek_treatment)
            * Gender mapped to numeric and One-Hot encoded
            * year_of_study extracted from "Year X" pattern via regex
            * cgpa_range parsed to pandas.Interval, midpoint stored as final_cgpa
            * student_id generated with "mental_" prefix
            * Source column added: "mental"

    Cleaned Student:
        - Shape: 5000 rows x 14 columns
        - Missing Values: 0
        - Changes Applied:
            * All column names converted to lowercase
            * Column renames: attendance_pct -> attendance,
              sleep_hours -> sleep_hours_per_day
            * Gender mapped and One-Hot encoded
            * student_id prefixed with "student_"
            * Source column added: "student"

## Data Integrity Checks

    - Confirmed zero missing values across all three cleaned datasets.
    - Verified that gender encoding produces only values {0, 1, 2} with correct One-Hot mapping.
    - Checked that student_id values are unique within each dataset.
    - Confirmed that the source column correctly identifies the origin of each record.


# =========================
# STAGE 2: FEATURE ENGINEERING VALIDATION
# =========================

## Feature Generation Results

After running `features.py`, the following feature-enhanced datasets were produced:

    Final Gaming Features:
        - Shape: 8000 rows x 24 columns (4 new features added)
        - Missing Values: 0
        - Infinite Values: 0
        - New Features:
            * gaming_intensity = gaming_hours / 24
                Mean: 0.1702, Std: 0.0962, Min: 0.0000, Max: 0.3333
            * academic_pressure_index = study_hours_per_day / (gaming_hours + 1)
                Mean: 1.7319, Std: 1.7375, Min: 0.0000, Max: 7.9200
            * sleep_gaming_balance = sleep_hours_per_day / (gaming_hours + 1)
                Mean: 3.6076, Std: 3.3193, Min: 0.0072, Max: 16.1000
            * total_screen_time = social_hours_day + gaming_hours
                Mean: 6.5936, Std: 2.7301, Min: 0.1300, Max: 12.9600
        - Division by zero was prevented by adding +1 to the denominator
          in academic_pressure_index and sleep_gaming_balance calculations.

    Final Mental Features:
        - Shape: 100 rows x 17 columns (1 new feature added)
        - Missing Values: 0
        - Infinite Values: 0
        - New Features:
            * total_mental_risk = depression + anxiety + panic_attack
                Mean: 1.02, Std: 0.97, Min: 0, Max: 3
                Distribution: ranges from 0 (no mental health risk indicators)
                to 3 (all three indicators present)

    Final Student Features:
        - Shape: 5000 rows x 16 columns (2 new features added)
        - Missing Values: 0
        - Infinite Values: 0
        - New Features:
            * study_efficiency = study_hours_per_day / (social_hours_week / 7 + 1)
                (only computed when social_hours_week column is available)
            * attendance_impact = (attendance * previous_cgpa) / 100
                Mean: 2.5940, Std: 0.5726, Min: 0.8411, Max: 4.0000

## Feature Validation Summary

    - All engineered features were confirmed to contain no NaN values.
    - All engineered features were confirmed to contain no Infinity values.
    - The +1 offset in denominator-based features (academic_pressure_index,
      sleep_gaming_balance, study_efficiency) successfully prevents
      division-by-zero errors.
    - All feature value ranges are within logically expected boundaries.


# =========================
# STAGE 3: DATA LEAKAGE VERIFICATION
# =========================

These checks were performed to ensure no data leakage exists in the pipeline:

    1. Target Variable Isolation:
       The `prepare_data()` function in `main.py` explicitly drops both possible
       target columns (`final_cgpa` and `grades`) from the feature set before training.
       This ensures the target variable does not appear as a feature.

    2. Leakage-Prone Column Removal:
       The columns `student_id`, `source`, and `cgpa_range` are all dropped.
       - student_id: Would create a unique identifier per row (perfect but meaningless predictor).
       - source: Would leak dataset origin information.
       - cgpa_range: Would directly encode the target variable.

    3. Scaling Inside Pipeline:
       StandardScaler is applied inside a sklearn Pipeline object, which means
       scaling parameters (mean, std) are fitted only on training data and then
       applied to test data. This prevents information from the test set from
       leaking into the training process.

    4. Train/Test Split:
       An 80/20 split with random_state=42 is used, applied before any model
       training or hyperparameter tuning.

    5. Cross-Validation:
       GridSearchCV with cv=5 is used, ensuring that the cross-validation folds
       are created within the training set only.


# =========================
# STAGE 4: MODEL TRAINING AND HYPERPARAMETER TUNING RESULTS
# =========================

## Gaming Dataset

    Dataset Split: X_train (6400, 14), X_test (1600, 14)
    Feature Count: 14

    Best Hyperparameters Found:
        Linear Regression: {} (no hyperparameters)
        Decision Tree: max_depth=10, min_samples_split=5
        Random Forest: n_estimators=100, max_depth=None, min_samples_split=2
        SVR: C=10, kernel=rbf
        KNN: n_neighbors=10, weights=distance
        Gradient Boosting: n_estimators=100, learning_rate=0.1, max_depth=5

    Cross-Validation R2 Scores:
        Linear Regression: 0.9255
        Decision Tree:     0.8869
        Random Forest:     0.9316
        SVR:               0.9283
        KNN:               0.8497
        Gradient Boosting: 0.9352 (best)

## Mental Health Dataset

    Dataset Split: X_train (80, 60), X_test (20, 60)
    Feature Count: 60

    Best Hyperparameters Found:
        Linear Regression: {} (no hyperparameters)
        Decision Tree: max_depth=5, min_samples_split=10
        Random Forest: n_estimators=100, max_depth=10, min_samples_split=5
        SVR: C=1, kernel=rbf
        KNN: n_neighbors=5, weights=uniform
        Gradient Boosting: n_estimators=50, learning_rate=0.01, max_depth=5

    Cross-Validation R2 Scores:
        Linear Regression: -3.3127
        Decision Tree:     -0.2679
        Random Forest:     -0.2235
        SVR:               -0.0792 (best)
        KNN:               -0.0857
        Gradient Boosting: -0.1373

    NOTE: All models produced negative R2 scores on this dataset. This indicates
    that the mental health dataset (only 100 samples with 60 features after
    One-Hot Encoding) does not contain sufficient predictive signal for accurate
    CGPA regression. The high dimensionality relative to sample size causes
    overfitting and poor generalization. This is an expected limitation and not
    a pipeline error.

## Student Performance Dataset

    Dataset Split: X_train (4000, 17), X_test (1000, 17)
    Feature Count: 17

    Best Hyperparameters Found:
        Linear Regression: {} (no hyperparameters)
        Decision Tree: max_depth=10, min_samples_split=10
        Random Forest: n_estimators=100, max_depth=10, min_samples_split=5
        SVR: C=1, kernel=rbf
        KNN: n_neighbors=7, weights=distance
        Gradient Boosting: n_estimators=100, learning_rate=0.1, max_depth=3

    Cross-Validation R2 Scores:
        Linear Regression: 0.9024
        Decision Tree:     0.8970
        Random Forest:     0.9310
        SVR:               0.9138
        KNN:               0.8064
        Gradient Boosting: 0.9419 (best)


# =========================
# STAGE 5: MODEL EVALUATION RESULTS (TEST SET)
# =========================

## Gaming Dataset - Test Set Performance

    | Model               | RMSE    | MAE     | R2      | Adjusted R2 |
    |---------------------|---------|---------|---------|-------------|
    | Linear Regression   | 6.5632  | 5.2327  | 0.9141  | 0.9129      |
    | Decision Tree       | 7.7472  | 6.0284  | 0.8803  | 0.8786      |
    | Random Forest       | 6.1882  | 4.7826  | 0.9236  | 0.9225      |
    | SVR                 | 6.2970  | 4.9120  | 0.9209  | 0.9198      |
    | KNN                 | 8.7623  | 6.8433  | 0.8468  | 0.8447      |
    | Gradient Boosting   | 5.9936  | 4.6847  | 0.9283  | 0.9273      |

    Best Model: Gradient Boosting (R2 = 0.9283, RMSE = 5.9936)
    Target Variable Range: 0.00 - 118.63 (mean = 66.18, std = 22.42)
    RMSE as Percentage of Target Range: 5.05%
    RMSE as Percentage of Target Std: 26.73%

    Observations:
    - All models achieve R2 > 0.84, indicating strong predictive capability.
    - Gradient Boosting provides the best balance of low RMSE and high R2.
    - KNN and Decision Tree show relatively weaker performance, likely due to
      high variance and sensitivity to local patterns.
    - The difference between CV R2 (0.9352) and test R2 (0.9283) for
      Gradient Boosting is small (0.0069), confirming no significant overfitting.

## Mental Health Dataset - Test Set Performance

    | Model               | RMSE    | MAE     | R2      | Adjusted R2 |
    |---------------------|---------|---------|---------|-------------|
    | Linear Regression   | 0.7366  | 0.5895  | -0.4282 | NaN         |
    | Decision Tree       | 0.6209  | 0.3651  | -0.0146 | NaN         |
    | Random Forest       | 0.6088  | 0.3907  | 0.0245  | NaN         |
    | SVR                 | 0.6455  | 0.3963  | -0.0966 | NaN         |
    | KNN                 | 0.6588  | 0.4148  | -0.1424 | NaN         |
    | Gradient Boosting   | 0.6123  | 0.3821  | 0.0132  | NaN         |

    Best Model: Random Forest (R2 = 0.0245, RMSE = 0.6088)
    Target Variable Range: 0.995 - 3.750 (mean = 3.38, std = 0.54)
    RMSE as Percentage of Target Range: 22.11%
    RMSE as Percentage of Target Std: 112.74%

    Adjusted R2 is NaN because the sample size (n=20 test samples) is not
    sufficiently larger than the feature count (p=60) to compute a valid
    adjusted R2 value (requires n > p + 1).

    Observations:
    - All R2 values are near or below zero. The models fail to outperform
      a simple mean-based baseline predictor.
    - Root cause: only 100 samples with 60 One-Hot encoded features creates
      a severely underdetermined system.
    - The CGPA target in this dataset has very low variance (most values cluster
      around 3.245 and 3.750), reducing the signal available for learning.
    - This is a dataset limitation, not a pipeline defect.

## Student Performance Dataset - Test Set Performance

    | Model               | RMSE    | MAE     | R2      | Adjusted R2 |
    |---------------------|---------|---------|---------|-------------|
    | Linear Regression   | 0.1588  | 0.1286  | 0.9121  | 0.9106      |
    | Decision Tree       | 0.1580  | 0.1209  | 0.9129  | 0.9114      |
    | Random Forest       | 0.1297  | 0.1045  | 0.9413  | 0.9403      |
    | SVR                 | 0.1434  | 0.1170  | 0.9283  | 0.9270      |
    | KNN                 | 0.2216  | 0.1747  | 0.8288  | 0.8259      |
    | Gradient Boosting   | 0.1172  | 0.0976  | 0.9521  | 0.9513      |

    Best Model: Gradient Boosting (R2 = 0.9521, RMSE = 0.1172)
    Target Variable Range: 1.16 - 4.00 (mean = 3.27, std = 0.51)
    RMSE as Percentage of Target Range: 4.13%
    RMSE as Percentage of Target Std: 22.98%

    Observations:
    - All models (except KNN) achieve R2 > 0.91, indicating excellent fit.
    - Gradient Boosting is the top performer with R2 = 0.9521.
    - The gap between CV R2 (0.9419) and test R2 (0.9521) is minimal and
      slightly positive, which confirms the model generalizes well.
    - KNN underperforms relative to other models (R2 = 0.8288), consistent
      with the curse of dimensionality and distance-based limitations.


# =========================
# STAGE 6: CROSS-DATASET COMPARISON
# =========================

## Best Model Comparison Across Datasets

    | Dataset  | Best Model        | Test R2  | CV R2   | RMSE    | CV-Test Gap |
    |----------|-------------------|----------|---------|---------|-------------|
    | Gaming   | Gradient Boosting | 0.9283   | 0.9352  | 5.9936  | +0.0069     |
    | Mental   | Random Forest     | 0.0245   | -0.2235 | 0.6088  | -0.2480     |
    | Student  | Gradient Boosting | 0.9521   | 0.9419  | 0.1172  | -0.0102     |

    Key Findings:
    - Gradient Boosting consistently outperforms other algorithms on Gaming
      and Student datasets.
    - The Mental Health dataset fails to produce meaningful predictions
      regardless of the algorithm used.
    - The Gaming dataset benefits from a large sample size (8000) and clear
      numeric features (gaming hours, study hours, sleep hours).
    - The Student dataset achieves the highest R2 overall (0.9521) with a
      well-balanced feature-to-sample ratio (17 features, 5000 samples).

## Model Ranking Consistency

    Gaming Dataset Ranking (by Test R2):
        1. Gradient Boosting (0.9283)
        2. Random Forest (0.9236)
        3. SVR (0.9209)
        4. Linear Regression (0.9141)
        5. Decision Tree (0.8803)
        6. KNN (0.8468)

    Student Dataset Ranking (by Test R2):
        1. Gradient Boosting (0.9521)
        2. Random Forest (0.9413)
        3. SVR (0.9283)
        4. Decision Tree (0.9129)
        5. Linear Regression (0.9121)
        6. KNN (0.8288)

    The top-3 ranking is identical across both viable datasets (Gaming and Student):
    Gradient Boosting > Random Forest > SVR. This consistency validates that the
    pipeline produces stable and reliable results regardless of the input dataset.


# =========================
# STAGE 7: FEATURE IMPORTANCE ANALYSIS
# =========================

## Gaming Dataset - Gradient Boosting Feature Importance

    | Rank | Feature                  | Importance |
    |------|--------------------------|------------|
    | 1    | academic_pressure_index  | 0.7831     |
    | 2    | sleep_gaming_balance     | 0.0720     |
    | 3    | study_hours_per_day      | 0.0696     |
    | 4    | sleep_hours_per_day      | 0.0336     |
    | 5    | attendance               | 0.0265     |
    | 6    | gaming_intensity         | 0.0044     |
    | 7    | gaming_hours             | 0.0038     |
    | 8    | reaction_time_ms         | 0.0015     |
    | 9    | device_usage             | 0.0010     |
    | 10   | addiction_score           | 0.0010     |

    Key Insight: The engineered feature `academic_pressure_index` (study_hours / gaming_hours + 1)
    dominates with 78.3% importance. This confirms that the ratio between study and gaming
    time is the single most powerful predictor of academic grades in the gaming dataset.
    Raw gaming_hours alone contributes only 0.38%, demonstrating the value of feature engineering.

## Mental Health Dataset - Random Forest Feature Importance

    | Rank | Feature                  | Importance |
    |------|--------------------------|------------|
    | 1    | course_Fiqh              | 0.2901     |
    | 2    | course_Biomedical science| 0.1798     |
    | 3    | age                      | 0.1122     |
    | 4    | year_of_study            | 0.0611     |
    | 5    | course_Engineering       | 0.0548     |
    | 6    | course_Human Resources   | 0.0339     |
    | 7    | course_BIT               | 0.0307     |
    | 8    | course_BCS               | 0.0290     |
    | 9    | panic_attack             | 0.0266     |
    | 10   | seek_treatment           | 0.0262     |

    Key Insight: Course-related One-Hot encoded features dominate the importance ranking.
    This suggests that academic department (course) is more predictive of CGPA than
    mental health indicators in this small dataset. However, given the poor overall model
    performance (R2 = 0.0245), these importance values should be interpreted with caution.

## Student Performance Dataset - Gradient Boosting Feature Importance

    | Rank | Feature                  | Importance |
    |------|--------------------------|------------|
    | 1    | previous_cgpa            | 0.7225     |
    | 2    | attendance_impact        | 0.1640     |
    | 3    | study_hours_per_day      | 0.0471     |
    | 4    | attendance               | 0.0415     |
    | 5    | study_efficiency         | 0.0128     |
    | 6    | sleep_hours_per_day      | 0.0077     |
    | 7    | social_hours_week        | 0.0041     |
    | 8    | age                      | 0.0002     |
    | 9    | major_Computer Science   | 0.0001     |
    | 10   | major_Engineering        | 0.0000     |

    Key Insight: `previous_cgpa` is by far the strongest predictor (72.2%) of final CGPA.
    The engineered feature `attendance_impact` (attendance * previous_cgpa / 100) ranks
    second at 16.4%, demonstrating that the interaction between attendance and prior
    academic performance is highly informative. Together, these two features account for
    88.7% of the model's decision-making.


# =========================
# STAGE 8: EXPLAINABILITY (SHAP) VERIFICATION
# =========================

SHAP (SHapley Additive exPlanations) analysis was executed for each dataset's best model.

    Gaming (Gradient Boosting):
        - TreeExplainer was used (optimal for tree-based models).
        - 300 samples were randomly selected from training data for computation.
        - SHAP summary plot saved to: qa_debug_and_test_outputs/reports/gaming_-_gradient_boosting_shap_summary.png
        - Execution completed without errors.

    Mental (Random Forest):
        - TreeExplainer was used.
        - Full training set (80 samples, under 300 limit) was used.
        - SHAP summary plot saved to: qa_debug_and_test_outputs/reports/mental_-_random_forest_shap_summary.png
        - Execution completed without errors.

    Student (Gradient Boosting):
        - TreeExplainer was used.
        - 300 samples were randomly selected from training data for computation.
        - SHAP summary plot saved to: qa_debug_and_test_outputs/reports/student_-_gradient_boosting_shap_summary.png
        - Execution completed without errors.

    Verification Notes:
    - The explainability module correctly identifies model type and selects the
      appropriate SHAP explainer (TreeExplainer for tree-based, LinearExplainer
      for linear models).
    - KernelExplainer is intentionally skipped for SVR and KNN models due to
      excessive computation time, and this is properly logged.
    - Sample limiting (max 300) prevents memory and runtime issues on large datasets.


# =========================
# STAGE 9: OUTPUT FILE VERIFICATION
# =========================

## Generated Output Files

All expected output files were confirmed to exist with non-zero file sizes:

    qa_debug_and_test_outputs/reports/ Directory (21 files):

        Metrics and Results (CSV):
            - Gaming_all_model_results.csv (546 bytes)
            - Gaming_metrics.csv (546 bytes)
            - Mental_all_model_results.csv (451 bytes)
            - Mental_metrics.csv (451 bytes)
            - Student_all_model_results.csv (567 bytes)
            - Student_metrics.csv (567 bytes)

        Model Comparison Plots (PNG):
            - Gaming_model_comparison.png (189,867 bytes)
            - Mental_model_comparison.png (177,507 bytes)
            - Student_model_comparison.png (181,405 bytes)

        Actual vs Predicted Plots (PNG):
            - Gaming_actual_vs_predicted.png (2,306,340 bytes)
            - Mental_actual_vs_predicted.png (334,385 bytes)
            - Student_actual_vs_predicted.png (1,755,954 bytes)

        Feature Importance (CSV + PNG):
            - gaming_-_gradient_boosting_feature_importance.csv (411 bytes)
            - gaming_-_gradient_boosting_feature_importance.png (143,739 bytes)
            - mental_-_random_forest_feature_importance.csv (381 bytes)
            - mental_-_random_forest_feature_importance.png (130,774 bytes)
            - student_-_gradient_boosting_feature_importance.csv (405 bytes)
            - student_-_gradient_boosting_feature_importance.png (139,776 bytes)

        SHAP Summary Plots (PNG):
            - gaming_-_gradient_boosting_shap_summary.png (413,257 bytes)
            - mental_-_random_forest_shap_summary.png (281,889 bytes)
            - student_-_gradient_boosting_shap_summary.png (346,767 bytes)

    data/processed/ Directory (6 files):

        Cleaned Datasets:
            - cleaned_gaming.csv (860,326 bytes)
            - cleaned_mental.csv (7,027 bytes)
            - cleaned_student.csv (364,937 bytes)

        Feature-Enhanced Datasets:
            - final_gaming_features.csv (1,329,865 bytes)
            - final_mental_features.csv (7,245 bytes)
            - final_student_features.csv (492,655 bytes)


# =========================
# STAGE 10: END-TO-END PIPELINE EXECUTION
# =========================

The full pipeline was executed in the following order:

    Step 1: python src/load_and_clean.py
        - Status: PASSED
        - Output: 3 cleaned CSV files generated in data/processed/
        - No errors or warnings.

    Step 2: python src/features.py
        - Status: PASSED
        - Output: 3 feature-enhanced CSV files generated in data/processed/
        - Console output confirmed:
            "Gaming features successfully built."
            "Mental features successfully built."
            "Student features successfully built."
            "[SUCCESS] All features have been generated and saved to 'data/processed'."

    Step 3: python src/main.py
        - Status: PASSED
        - Output: 21 files generated in qa_debug_and_test_outputs/reports/
        - All three datasets processed sequentially (Gaming, Mental, Student).
        - All 6 models trained with GridSearchCV for each dataset (18 total training runs).
        - All evaluation metrics computed and saved.
        - All visualization plots generated and saved.
        - All explainability analyses (Feature Importance + SHAP) completed.
        - No exceptions, no warnings, no data corruption detected.


# =========================
# STAGE 11: DEPENDENCY AND ENVIRONMENT VERIFICATION
# =========================

## requirements.txt Content

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    shap
    scipy
    joblib

## Verification

    - All listed packages were successfully installed and imported.
    - No import errors occurred during any pipeline stage.
    - No version conflicts were detected between dependencies.
    - The `threadpoolctl` package (required by scikit-learn) was automatically
      installed as a transitive dependency.


# =========================
# STAGE 12: AUTOMATED TEST SCRIPTS
# =========================

Two automated scripts were developed to enable repeatable QA validation
and full pipeline re-execution without manual intervention.

## run_all.py — End-to-End Pipeline Executor

    Purpose:
        Executes the entire data science pipeline in the correct sequential order.
        Provides pass/fail status for each step and an overall execution summary.

    Execution Steps:
        Step 1: Data Loading and Cleaning   (src/load_and_clean.py)
        Step 2: Feature Engineering          (src/features.py)
        Step 3: Model Training, Evaluation,  (src/main.py)
                and Explainability

    Usage:
        python qa_debug_and_test_outputs/run_all.py

    Execution Result (2026-05-25):
        [PASS] Step 1: Data Loading and Cleaning
        [PASS] Step 2: Feature Engineering
        [PASS] Step 3: Model Training, Evaluation, and Explainability

        Result: 3/3 steps passed.
        Total execution time: 108.38 seconds.
        Pipeline completed successfully.

    Key Features:
        - Automatically detects project root and script paths.
        - Stops execution immediately if any step fails ([ABORT] mechanism).
        - Reports per-step timing and overall pipeline duration.
        - Captures both stdout and stderr for debugging on failure.

## test_pipeline.py — Automated QA Test Suite (pytest)

    Purpose:
        Comprehensive automated test suite that validates every stage
        of the data science pipeline. Covers data integrity, feature
        engineering correctness, data leakage prevention, model structure
        verification, evaluation metric validation, and output file checks.

    Usage:
        pytest qa_debug_and_test_outputs/test_pipeline.py -v

    Execution Result (2026-05-25):
        89 tests passed, 0 failed, 0 skipped.
        Execution time: 12.56 seconds.

    Test Groups (10 groups, 89 tests total):

        Group 1 — Raw Data Existence (6 tests):
            Verifies that gaming.csv, mental.csv, and student.csv exist
            in data/raw/ and are not empty.

        Group 2 — Data Cleaning Validation (18 tests):
            Checks that cleaned datasets have zero missing values,
            contain all required columns (student_id, final_cgpa, gender,
            is_female, is_male, source), have valid row counts, correct
            source tags, proper gender encoding ({0, 1, 2}), and
            correct student_id prefixes (gaming_, mental_, student_).

        Group 3 — Feature Engineering Validation (13 tests):
            Confirms that feature-enhanced datasets contain no NaN or
            Infinity values, verifies existence of engineered features
            (gaming_intensity, academic_pressure_index, sleep_gaming_balance,
            total_screen_time, total_mental_risk, attendance_impact), and
            validates that feature value ranges are within expected boundaries.

        Group 4 — Data Leakage Prevention (6 tests):
            Tests that prepare_data() correctly drops target variables
            (grades, final_cgpa), identifier columns (student_id),
            source column, and cgpa_range. Confirms target isolation
            and correct target variable length/name.

        Group 5 — Model Pipeline Structure (5 tests):
            Verifies that all 6 regression models are defined, each
            pipeline contains a StandardScaler step and a model step,
            parameter grids match pipeline definitions, and the scaler
            is specifically an instance of StandardScaler.

        Group 6 — Evaluation Metrics Validation (4 tests):
            Tests calculate_metrics() with synthetic data: verifies
            all metric keys (RMSE, MAE, R2, Adjusted R2) are returned,
            perfect predictions yield R2=1.0 and RMSE=0.0, and RMSE/MAE
            are non-negative.

        Group 7 — Output File Generation (15 tests):
            Checks existence and non-zero size of all output files:
            3 metrics CSVs, 3 model comparison plots, 3 actual vs
            predicted plots, feature importance files (CSV + PNG),
            3 SHAP summary plots, and total file count (21 files
            in reports directory).

        Group 8 — Metrics Content Validation (8 tests):
            Validates that saved metrics CSVs contain 6 model rows,
            Gaming and Student R2 values are positive, RMSE values
            are within reasonable bounds (Gaming < 50, Student < 2.0),
            best model R2 exceeds thresholds (Gaming > 0.85,
            Student > 0.90), and all expected metric columns exist.

        Group 9 — Processed Data Integrity (5 tests):
            Verifies processed directory contains 6 CSV files,
            cleaned dataset shapes match expectations (Gaming: 8000,
            Mental: 100, Student: 5000), and feature-enhanced files
            have more columns than cleaned files.

        Group 10 — Dependency Verification (8 tests):
            Confirms that all required packages can be imported:
            pandas, numpy, sklearn, matplotlib, seaborn, shap,
            scipy, joblib.

    Full Test Results (all 89 PASSED):

        TestRawDataExists::test_gaming_csv_exists .................. PASSED
        TestRawDataExists::test_mental_csv_exists .................. PASSED
        TestRawDataExists::test_student_csv_exists ................. PASSED
        TestRawDataExists::test_gaming_csv_not_empty ............... PASSED
        TestRawDataExists::test_mental_csv_not_empty ............... PASSED
        TestRawDataExists::test_student_csv_not_empty .............. PASSED
        TestDataCleaning::test_gaming_no_missing_values ............ PASSED
        TestDataCleaning::test_mental_no_missing_values ............ PASSED
        TestDataCleaning::test_student_no_missing_values ........... PASSED
        TestDataCleaning::test_gaming_has_required_columns ......... PASSED
        TestDataCleaning::test_mental_has_required_columns ......... PASSED
        TestDataCleaning::test_student_has_required_columns ........ PASSED
        TestDataCleaning::test_gaming_row_count .................... PASSED
        TestDataCleaning::test_mental_row_count .................... PASSED
        TestDataCleaning::test_student_row_count ................... PASSED
        TestDataCleaning::test_gaming_source_tag ................... PASSED
        TestDataCleaning::test_mental_source_tag ................... PASSED
        TestDataCleaning::test_student_source_tag .................. PASSED
        TestDataCleaning::test_gaming_gender_values ................ PASSED
        TestDataCleaning::test_mental_gender_values ................ PASSED
        TestDataCleaning::test_student_gender_values ............... PASSED
        TestDataCleaning::test_gaming_id_prefix .................... PASSED
        TestDataCleaning::test_mental_id_prefix .................... PASSED
        TestDataCleaning::test_student_id_prefix ................... PASSED
        TestFeatureEngineering::test_gaming_features_no_nan ........ PASSED
        TestFeatureEngineering::test_mental_features_no_nan ........ PASSED
        TestFeatureEngineering::test_student_features_no_nan ....... PASSED
        TestFeatureEngineering::test_gaming_features_no_infinity ... PASSED
        TestFeatureEngineering::test_mental_features_no_infinity ... PASSED
        TestFeatureEngineering::test_student_features_no_infinity .. PASSED
        TestFeatureEngineering::test_gaming_has_engineered_features  PASSED
        TestFeatureEngineering::test_mental_has_engineered_features  PASSED
        TestFeatureEngineering::test_student_has_engineered_features PASSED
        TestFeatureEngineering::test_gaming_intensity_range ........ PASSED
        TestFeatureEngineering::test_mental_risk_range ............. PASSED
        TestFeatureEngineering::test_academic_pressure_non_negative  PASSED
        TestFeatureEngineering::test_attendance_impact_non_negative  PASSED
        TestDataLeakage::test_prepare_data_drops_targets ........... PASSED
        TestDataLeakage::test_prepare_data_drops_student_id ........ PASSED
        TestDataLeakage::test_prepare_data_drops_source ............ PASSED
        TestDataLeakage::test_prepare_data_drops_cgpa_range ........ PASSED
        TestDataLeakage::test_target_not_in_feature_set ............ PASSED
        TestDataLeakage::test_target_values_correct ................ PASSED
        TestModelPipeline::test_all_six_models_defined ............. PASSED
        TestModelPipeline::test_all_pipelines_have_scaler .......... PASSED
        TestModelPipeline::test_all_pipelines_have_model_step ...... PASSED
        TestModelPipeline::test_param_grids_match_pipelines ........ PASSED
        TestModelPipeline::test_scaler_is_standard_scaler .......... PASSED
        TestEvaluationMetrics::test_calculate_metrics_returns_all_keys PASSED
        TestEvaluationMetrics::test_perfect_prediction_r2 .......... PASSED
        TestEvaluationMetrics::test_rmse_is_non_negative ........... PASSED
        TestEvaluationMetrics::test_mae_is_non_negative ............ PASSED
        TestOutputFiles::test_gaming_metrics_csv_exists ............ PASSED
        TestOutputFiles::test_mental_metrics_csv_exists ............ PASSED
        TestOutputFiles::test_student_metrics_csv_exists ........... PASSED
        TestOutputFiles::test_gaming_comparison_plot_exists ......... PASSED
        TestOutputFiles::test_mental_comparison_plot_exists ......... PASSED
        TestOutputFiles::test_student_comparison_plot_exists ........ PASSED
        TestOutputFiles::test_gaming_actual_vs_predicted_exists ..... PASSED
        TestOutputFiles::test_mental_actual_vs_predicted_exists ..... PASSED
        TestOutputFiles::test_student_actual_vs_predicted_exists .... PASSED
        TestOutputFiles::test_gaming_feature_importance_csv_exists .. PASSED
        TestOutputFiles::test_gaming_feature_importance_plot_exists . PASSED
        TestOutputFiles::test_gaming_shap_plot_exists ............... PASSED
        TestOutputFiles::test_mental_shap_plot_exists ............... PASSED
        TestOutputFiles::test_student_shap_plot_exists .............. PASSED
        TestOutputFiles::test_reports_directory_has_21_files ........ PASSED
        TestMetricsContent::test_gaming_has_six_models ............. PASSED
        TestMetricsContent::test_student_has_six_models ............ PASSED
        TestMetricsContent::test_gaming_r2_positive ................ PASSED
        TestMetricsContent::test_student_r2_positive ............... PASSED
        TestMetricsContent::test_gaming_rmse_reasonable ............ PASSED
        TestMetricsContent::test_student_rmse_reasonable ........... PASSED
        TestMetricsContent::test_gaming_best_model_r2_above_threshold PASSED
        TestMetricsContent::test_student_best_model_r2_above_threshold PASSED
        TestMetricsContent::test_metrics_columns_complete .......... PASSED
        TestProcessedDataIntegrity::test_processed_directory_has_6_files PASSED
        TestProcessedDataIntegrity::test_cleaned_gaming_shape ...... PASSED
        TestProcessedDataIntegrity::test_cleaned_mental_shape ...... PASSED
        TestProcessedDataIntegrity::test_cleaned_student_shape ..... PASSED
        TestProcessedDataIntegrity::test_feature_files_have_more_columns PASSED
        TestDependencies::test_import_pandas ....................... PASSED
        TestDependencies::test_import_numpy ....................... PASSED
        TestDependencies::test_import_sklearn ..................... PASSED
        TestDependencies::test_import_matplotlib .................. PASSED
        TestDependencies::test_import_seaborn ..................... PASSED
        TestDependencies::test_import_shap ........................ PASSED
        TestDependencies::test_import_scipy ....................... PASSED
        TestDependencies::test_import_joblib ...................... PASSED

        ============================= 89 passed in 12.56s =============================


# =========================
# CONCLUSION
# =========================

The complete pipeline was tested and validated across all stages: data loading,
cleaning, feature engineering, model training, evaluation, and explainability.

    - Gaming Dataset: Pipeline performs excellently (best R2 = 0.9283).
    - Student Dataset: Pipeline performs excellently (best R2 = 0.9521).
    - Mental Dataset: Poor performance due to insufficient data, not pipeline errors.

All 27 output files (6 CSV + 21 reports) were generated and collected in qa_debug_and_test_outputs/.
No code-level bugs, data leakage, or pipeline breaks were detected.
The pipeline is confirmed to be production-ready for the Gaming and Student datasets.

## Automated Validation Summary

    run_all.py:
        - 3/3 pipeline steps PASSED in 108.38 seconds.
        - Full end-to-end pipeline is reproducible with a single command.

    test_pipeline.py:
        - 89/89 automated tests PASSED in 12.56 seconds.
        - All 10 test groups (data existence, cleaning, features, leakage,
          model structure, metrics, output files, metrics content,
          processed data integrity, dependencies) completed without failure.
