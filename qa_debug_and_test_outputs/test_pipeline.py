"""
# =========================
# TEST PIPELINE - AUTOMATED QA TESTS
# =========================

Automated test suite for validating the entire data science pipeline.
Covers data integrity, feature engineering, data leakage prevention,
model output verification, and output file generation.

Usage:
    pytest qa_debug_and_test_outputs/test_pipeline.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# =========================
# PATH CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

sys.path.insert(0, SRC_DIR)


# =========================
# TEST GROUP 1: RAW DATA EXISTENCE
# =========================

class TestRawDataExists:
    """
    Verifies that all required raw data files
    are present before the pipeline runs.
    """

    def test_gaming_csv_exists(self):
        path = os.path.join(RAW_DATA_DIR, "gaming.csv")
        assert os.path.exists(path), (
            f"Raw gaming dataset not found at {path}"
        )

    def test_mental_csv_exists(self):
        path = os.path.join(RAW_DATA_DIR, "mental.csv")
        assert os.path.exists(path), (
            f"Raw mental dataset not found at {path}"
        )

    def test_student_csv_exists(self):
        path = os.path.join(RAW_DATA_DIR, "student.csv")
        assert os.path.exists(path), (
            f"Raw student dataset not found at {path}"
        )

    def test_gaming_csv_not_empty(self):
        path = os.path.join(RAW_DATA_DIR, "gaming.csv")
        assert os.path.getsize(path) > 0, (
            "gaming.csv file is empty"
        )

    def test_mental_csv_not_empty(self):
        path = os.path.join(RAW_DATA_DIR, "mental.csv")
        assert os.path.getsize(path) > 0, (
            "mental.csv file is empty"
        )

    def test_student_csv_not_empty(self):
        path = os.path.join(RAW_DATA_DIR, "student.csv")
        assert os.path.getsize(path) > 0, (
            "student.csv file is empty"
        )


# =========================
# TEST GROUP 2: DATA CLEANING VALIDATION
# =========================

class TestDataCleaning:
    """
    Verifies that load_and_clean.py produced
    valid cleaned datasets with no missing values.
    """

    @pytest.fixture
    def cleaned_gaming(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_gaming.csv"
        )
        assert os.path.exists(path), (
            f"Cleaned gaming file not found at {path}"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def cleaned_mental(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_mental.csv"
        )
        assert os.path.exists(path), (
            f"Cleaned mental file not found at {path}"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def cleaned_student(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_student.csv"
        )
        assert os.path.exists(path), (
            f"Cleaned student file not found at {path}"
        )
        return pd.read_csv(path)

    # --- Missing Values ---

    def test_gaming_no_missing_values(self, cleaned_gaming):
        missing = cleaned_gaming.isnull().sum().sum()
        assert missing == 0, (
            f"Gaming dataset has {missing} missing values"
        )

    def test_mental_no_missing_values(self, cleaned_mental):
        missing = cleaned_mental.isnull().sum().sum()
        assert missing == 0, (
            f"Mental dataset has {missing} missing values"
        )

    def test_student_no_missing_values(self, cleaned_student):
        missing = cleaned_student.isnull().sum().sum()
        assert missing == 0, (
            f"Student dataset has {missing} missing values"
        )

    # --- Required Columns ---

    def test_gaming_has_required_columns(self, cleaned_gaming):
        required = [
            "student_id", "grades", "final_cgpa",
            "is_female", "is_male", "source"
        ]
        for col in required:
            assert col in cleaned_gaming.columns, (
                f"Gaming dataset missing column: {col}"
            )

    def test_mental_has_required_columns(self, cleaned_mental):
        required = [
            "student_id", "final_cgpa",
            "is_female", "is_male", "source"
        ]
        for col in required:
            assert col in cleaned_mental.columns, (
                f"Mental dataset missing column: {col}"
            )

    def test_student_has_required_columns(self, cleaned_student):
        required = [
            "student_id", "final_cgpa",
            "is_female", "is_male", "source"
        ]
        for col in required:
            assert col in cleaned_student.columns, (
                f"Student dataset missing column: {col}"
            )

    # --- Row Count Validation ---

    def test_gaming_row_count(self, cleaned_gaming):
        assert len(cleaned_gaming) > 0, (
            "Gaming dataset has zero rows"
        )

    def test_mental_row_count(self, cleaned_mental):
        assert len(cleaned_mental) > 0, (
            "Mental dataset has zero rows"
        )

    def test_student_row_count(self, cleaned_student):
        assert len(cleaned_student) > 0, (
            "Student dataset has zero rows"
        )

    # --- Source Column Validation ---

    def test_gaming_source_tag(self, cleaned_gaming):
        assert (cleaned_gaming["source"] == "gaming").all(), (
            "Gaming dataset source column contains incorrect values"
        )

    def test_mental_source_tag(self, cleaned_mental):
        assert (cleaned_mental["source"] == "mental").all(), (
            "Mental dataset source column contains incorrect values"
        )

    def test_student_source_tag(self, cleaned_student):
        assert (cleaned_student["source"] == "student").all(), (
            "Student dataset source column contains incorrect values"
        )

    # --- Gender Encoding ---

    def test_gaming_gender_values(self, cleaned_gaming):
        valid_values = {0, 1, 2}
        actual_values = set(cleaned_gaming["gender"].unique())
        assert actual_values.issubset(valid_values), (
            f"Gaming gender has unexpected values: "
            f"{actual_values - valid_values}"
        )

    def test_mental_gender_values(self, cleaned_mental):
        valid_values = {0, 1, 2}
        actual_values = set(cleaned_mental["gender"].unique())
        assert actual_values.issubset(valid_values), (
            f"Mental gender has unexpected values: "
            f"{actual_values - valid_values}"
        )

    def test_student_gender_values(self, cleaned_student):
        valid_values = {0, 1, 2}
        actual_values = set(cleaned_student["gender"].unique())
        assert actual_values.issubset(valid_values), (
            f"Student gender has unexpected values: "
            f"{actual_values - valid_values}"
        )

    # --- Student ID Prefix ---

    def test_gaming_id_prefix(self, cleaned_gaming):
        assert cleaned_gaming["student_id"].str.startswith(
            "gaming_"
        ).all(), (
            "Gaming student_id values do not have 'gaming_' prefix"
        )

    def test_mental_id_prefix(self, cleaned_mental):
        assert cleaned_mental["student_id"].str.startswith(
            "mental_"
        ).all(), (
            "Mental student_id values do not have 'mental_' prefix"
        )

    def test_student_id_prefix(self, cleaned_student):
        assert cleaned_student["student_id"].str.startswith(
            "student_"
        ).all(), (
            "Student student_id values do not have 'student_' prefix"
        )


# =========================
# TEST GROUP 3: FEATURE ENGINEERING VALIDATION
# =========================

class TestFeatureEngineering:
    """
    Verifies that features.py produced valid
    feature-enhanced datasets with no NaN or
    Infinity values.
    """

    @pytest.fixture
    def gaming_features(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_gaming_features.csv"
        )
        assert os.path.exists(path), (
            f"Gaming features file not found at {path}"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def mental_features(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_mental_features.csv"
        )
        assert os.path.exists(path), (
            f"Mental features file not found at {path}"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def student_features(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_student_features.csv"
        )
        assert os.path.exists(path), (
            f"Student features file not found at {path}"
        )
        return pd.read_csv(path)

    # --- No NaN Values ---

    def test_gaming_features_no_nan(self, gaming_features):
        nan_count = gaming_features.isnull().sum().sum()
        assert nan_count == 0, (
            f"Gaming features contain {nan_count} NaN values"
        )

    def test_mental_features_no_nan(self, mental_features):
        nan_count = mental_features.isnull().sum().sum()
        assert nan_count == 0, (
            f"Mental features contain {nan_count} NaN values"
        )

    def test_student_features_no_nan(self, student_features):
        nan_count = student_features.isnull().sum().sum()
        assert nan_count == 0, (
            f"Student features contain {nan_count} NaN values"
        )

    # --- No Infinity Values ---

    def test_gaming_features_no_infinity(self, gaming_features):
        numeric_cols = gaming_features.select_dtypes(
            include=[np.number]
        )
        inf_count = np.isinf(numeric_cols).sum().sum()
        assert inf_count == 0, (
            f"Gaming features contain {inf_count} Infinity values"
        )

    def test_mental_features_no_infinity(self, mental_features):
        numeric_cols = mental_features.select_dtypes(
            include=[np.number]
        )
        inf_count = np.isinf(numeric_cols).sum().sum()
        assert inf_count == 0, (
            f"Mental features contain {inf_count} Infinity values"
        )

    def test_student_features_no_infinity(self, student_features):
        numeric_cols = student_features.select_dtypes(
            include=[np.number]
        )
        inf_count = np.isinf(numeric_cols).sum().sum()
        assert inf_count == 0, (
            f"Student features contain {inf_count} Infinity values"
        )

    # --- Engineered Feature Existence ---

    def test_gaming_has_engineered_features(self, gaming_features):
        engineered = [
            "gaming_intensity",
            "academic_pressure_index",
            "sleep_gaming_balance",
            "total_screen_time"
        ]
        for col in engineered:
            assert col in gaming_features.columns, (
                f"Gaming missing engineered feature: {col}"
            )

    def test_mental_has_engineered_features(self, mental_features):
        assert "total_mental_risk" in mental_features.columns, (
            "Mental missing engineered feature: total_mental_risk"
        )

    def test_student_has_engineered_features(self, student_features):
        assert "attendance_impact" in student_features.columns, (
            "Student missing engineered feature: attendance_impact"
        )

    # --- Feature Value Range Checks ---

    def test_gaming_intensity_range(self, gaming_features):
        assert gaming_features["gaming_intensity"].min() >= 0, (
            "gaming_intensity has negative values"
        )
        assert gaming_features["gaming_intensity"].max() <= 1.0, (
            "gaming_intensity exceeds 1.0 (hours/24)"
        )

    def test_mental_risk_range(self, mental_features):
        assert mental_features["total_mental_risk"].min() >= 0, (
            "total_mental_risk has negative values"
        )
        assert mental_features["total_mental_risk"].max() <= 3, (
            "total_mental_risk exceeds 3 "
            "(max sum of depression+anxiety+panic_attack)"
        )

    def test_academic_pressure_non_negative(self, gaming_features):
        assert gaming_features[
            "academic_pressure_index"
        ].min() >= 0, (
            "academic_pressure_index has negative values"
        )

    def test_attendance_impact_non_negative(self, student_features):
        assert student_features[
            "attendance_impact"
        ].min() >= 0, (
            "attendance_impact has negative values"
        )


# =========================
# TEST GROUP 4: DATA LEAKAGE PREVENTION
# =========================

class TestDataLeakage:
    """
    Verifies that the pipeline properly prevents
    data leakage by checking column handling in
    the prepare_data function.
    """

    @pytest.fixture
    def gaming_features(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_gaming_features.csv"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def student_features(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_student_features.csv"
        )
        return pd.read_csv(path)

    def test_prepare_data_drops_targets(self, gaming_features):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        X, y = prepare_data(gaming_features, "grades")

        assert "grades" not in X.columns, (
            "Target variable 'grades' was not dropped from features"
        )
        assert "final_cgpa" not in X.columns, (
            "Leakage column 'final_cgpa' was not dropped from features"
        )

    def test_prepare_data_drops_student_id(self, gaming_features):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        X, y = prepare_data(gaming_features, "grades")

        assert "student_id" not in X.columns, (
            "Leakage column 'student_id' was not dropped from features"
        )

    def test_prepare_data_drops_source(self, gaming_features):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        X, y = prepare_data(gaming_features, "grades")

        assert "source" not in X.columns, (
            "Leakage column 'source' was not dropped from features"
        )

    def test_prepare_data_drops_cgpa_range(self):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "final_mental_features.csv"
        )
        df = pd.read_csv(path)
        X, y = prepare_data(df, "final_cgpa")

        assert "cgpa_range" not in X.columns, (
            "Leakage column 'cgpa_range' was not dropped from features"
        )

    def test_target_not_in_feature_set(self, student_features):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        X, y = prepare_data(student_features, "final_cgpa")

        assert "final_cgpa" not in X.columns, (
            "Target variable found in feature set - data leakage detected"
        )

    def test_target_values_correct(self, gaming_features):
        from main import prepare_data  # type: ignore # noqa # pylint: disable=import-error
        X, y = prepare_data(gaming_features, "grades")

        assert len(y) == len(gaming_features), (
            "Target variable length does not match dataset length"
        )
        assert y.name == "grades", (
            f"Target variable name is '{y.name}', expected 'grades'"
        )


# =========================
# TEST GROUP 5: MODEL PIPELINE STRUCTURE
# =========================

class TestModelPipeline:
    """
    Verifies that models.py defines the correct
    pipeline structure with StandardScaler and
    proper hyperparameter grids.
    """

    def test_all_six_models_defined(self):
        from models import get_model_pipelines  # type: ignore # noqa # pylint: disable=import-error
        pipelines, param_grids = get_model_pipelines()

        expected_models = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "SVR",
            "KNN",
            "Gradient Boosting"
        ]

        for model_name in expected_models:
            assert model_name in pipelines, (
                f"Model '{model_name}' not found in pipelines"
            )

    def test_all_pipelines_have_scaler(self):
        from models import get_model_pipelines  # type: ignore # noqa # pylint: disable=import-error
        pipelines, _ = get_model_pipelines()

        for name, pipeline in pipelines.items():
            step_names = [
                step[0] for step in pipeline.steps
            ]
            assert "scaler" in step_names, (
                f"Pipeline '{name}' is missing StandardScaler step"
            )

    def test_all_pipelines_have_model_step(self):
        from models import get_model_pipelines  # type: ignore # noqa # pylint: disable=import-error
        pipelines, _ = get_model_pipelines()

        for name, pipeline in pipelines.items():
            step_names = [
                step[0] for step in pipeline.steps
            ]
            assert "model" in step_names, (
                f"Pipeline '{name}' is missing 'model' step"
            )

    def test_param_grids_match_pipelines(self):
        from models import get_model_pipelines  # type: ignore # noqa # pylint: disable=import-error
        pipelines, param_grids = get_model_pipelines()

        for name in pipelines:
            assert name in param_grids, (
                f"No parameter grid defined for '{name}'"
            )

    def test_scaler_is_standard_scaler(self):
        from models import get_model_pipelines  # type: ignore # noqa # pylint: disable=import-error
        from sklearn.preprocessing import StandardScaler
        pipelines, _ = get_model_pipelines()

        for name, pipeline in pipelines.items():
            scaler = pipeline.named_steps["scaler"]
            assert isinstance(scaler, StandardScaler), (
                f"Pipeline '{name}' scaler is not StandardScaler"
            )


# =========================
# TEST GROUP 6: EVALUATION METRICS VALIDATION
# =========================

class TestEvaluationMetrics:
    """
    Verifies that evaluation.py produces valid
    metric calculations.
    """

    def test_calculate_metrics_returns_all_keys(self):
        from evaluation import calculate_metrics  # type: ignore # noqa # pylint: disable=import-error

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        result = calculate_metrics(
            y_true, y_pred, (5, 2)
        )

        expected_keys = ["RMSE", "MAE", "R2", "Adjusted R2"]
        for key in expected_keys:
            assert key in result, (
                f"Missing metric key: {key}"
            )

    def test_perfect_prediction_r2(self):
        from evaluation import calculate_metrics  # type: ignore # noqa # pylint: disable=import-error

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_metrics(
            y_true, y_pred, (5, 2)
        )

        assert result["R2"] == pytest.approx(1.0), (
            "Perfect prediction should yield R2 = 1.0"
        )
        assert result["RMSE"] == pytest.approx(0.0), (
            "Perfect prediction should yield RMSE = 0.0"
        )
        assert result["MAE"] == pytest.approx(0.0), (
            "Perfect prediction should yield MAE = 0.0"
        )

    def test_rmse_is_non_negative(self):
        from evaluation import calculate_metrics  # type: ignore # noqa # pylint: disable=import-error

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        result = calculate_metrics(
            y_true, y_pred, (3, 1)
        )

        assert result["RMSE"] >= 0, (
            "RMSE should never be negative"
        )

    def test_mae_is_non_negative(self):
        from evaluation import calculate_metrics  # type: ignore # noqa # pylint: disable=import-error

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        result = calculate_metrics(
            y_true, y_pred, (3, 1)
        )

        assert result["MAE"] >= 0, (
            "MAE should never be negative"
        )


# =========================
# TEST GROUP 7: OUTPUT FILE GENERATION
# =========================

class TestOutputFiles:
    """
    Verifies that all expected output files
    were generated with non-zero file sizes.
    """

    # --- Metrics CSV Files ---

    def test_gaming_metrics_csv_exists(self):
        path = os.path.join(REPORTS_DIR, "Gaming_metrics.csv")
        assert os.path.exists(path), (
            f"Gaming metrics CSV not found at {path}"
        )
        assert os.path.getsize(path) > 0, (
            "Gaming metrics CSV is empty"
        )

    def test_mental_metrics_csv_exists(self):
        path = os.path.join(REPORTS_DIR, "Mental_metrics.csv")
        assert os.path.exists(path), (
            f"Mental metrics CSV not found at {path}"
        )
        assert os.path.getsize(path) > 0, (
            "Mental metrics CSV is empty"
        )

    def test_student_metrics_csv_exists(self):
        path = os.path.join(REPORTS_DIR, "Student_metrics.csv")
        assert os.path.exists(path), (
            f"Student metrics CSV not found at {path}"
        )
        assert os.path.getsize(path) > 0, (
            "Student metrics CSV is empty"
        )

    # --- Model Comparison Plots ---

    def test_gaming_comparison_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Gaming_model_comparison.png"
        )
        assert os.path.exists(path), (
            f"Gaming comparison plot not found at {path}"
        )
        assert os.path.getsize(path) > 1000, (
            "Gaming comparison plot is suspiciously small"
        )

    def test_mental_comparison_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Mental_model_comparison.png"
        )
        assert os.path.exists(path), (
            f"Mental comparison plot not found at {path}"
        )

    def test_student_comparison_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Student_model_comparison.png"
        )
        assert os.path.exists(path), (
            f"Student comparison plot not found at {path}"
        )

    # --- Actual vs Predicted Plots ---

    def test_gaming_actual_vs_predicted_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Gaming_actual_vs_predicted.png"
        )
        assert os.path.exists(path), (
            f"Gaming actual vs predicted plot not found at {path}"
        )

    def test_mental_actual_vs_predicted_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Mental_actual_vs_predicted.png"
        )
        assert os.path.exists(path), (
            f"Mental actual vs predicted plot not found at {path}"
        )

    def test_student_actual_vs_predicted_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "Student_actual_vs_predicted.png"
        )
        assert os.path.exists(path), (
            f"Student actual vs predicted plot not found at {path}"
        )

    # --- Feature Importance Files ---

    def test_gaming_feature_importance_csv_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "gaming_-_gradient_boosting_feature_importance.csv"
        )
        assert os.path.exists(path), (
            f"Gaming feature importance CSV not found at {path}"
        )

    def test_gaming_feature_importance_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "gaming_-_gradient_boosting_feature_importance.png"
        )
        assert os.path.exists(path), (
            f"Gaming feature importance plot not found at {path}"
        )

    # --- SHAP Summary Plots ---

    def test_gaming_shap_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "gaming_-_gradient_boosting_shap_summary.png"
        )
        assert os.path.exists(path), (
            f"Gaming SHAP summary plot not found at {path}"
        )

    def test_mental_shap_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "mental_-_random_forest_shap_summary.png"
        )
        assert os.path.exists(path), (
            f"Mental SHAP summary plot not found at {path}"
        )

    def test_student_shap_plot_exists(self):
        path = os.path.join(
            REPORTS_DIR,
            "student_-_gradient_boosting_shap_summary.png"
        )
        assert os.path.exists(path), (
            f"Student SHAP summary plot not found at {path}"
        )

    # --- Total File Count ---

    def test_reports_directory_has_21_files(self):
        if not os.path.exists(REPORTS_DIR):
            pytest.skip("Reports directory does not exist yet")

        file_count = len([
            f for f in os.listdir(REPORTS_DIR)
            if os.path.isfile(os.path.join(REPORTS_DIR, f))
        ])

        assert file_count == 21, (
            f"Expected 21 report files, found {file_count}"
        )


# =========================
# TEST GROUP 8: METRICS CONTENT VALIDATION
# =========================

class TestMetricsContent:
    """
    Verifies that the saved metrics CSV files
    contain valid and reasonable values.
    """

    @pytest.fixture
    def gaming_metrics(self):
        path = os.path.join(
            REPORTS_DIR,
            "Gaming_metrics.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Gaming metrics not generated yet")
        return pd.read_csv(path, index_col=0)

    @pytest.fixture
    def student_metrics(self):
        path = os.path.join(
            REPORTS_DIR,
            "Student_metrics.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Student metrics not generated yet")
        return pd.read_csv(path, index_col=0)

    def test_gaming_has_six_models(self, gaming_metrics):
        assert len(gaming_metrics) == 6, (
            f"Expected 6 model rows, found {len(gaming_metrics)}"
        )

    def test_student_has_six_models(self, student_metrics):
        assert len(student_metrics) == 6, (
            f"Expected 6 model rows, found {len(student_metrics)}"
        )

    def test_gaming_r2_positive(self, gaming_metrics):
        assert (gaming_metrics["R2"] > 0).all(), (
            "Some Gaming models have negative R2 values, "
            "which may indicate a pipeline issue for this dataset"
        )

    def test_student_r2_positive(self, student_metrics):
        assert (student_metrics["R2"] > 0).all(), (
            "Some Student models have negative R2 values, "
            "which may indicate a pipeline issue for this dataset"
        )

    def test_gaming_rmse_reasonable(self, gaming_metrics):
        assert (gaming_metrics["RMSE"] < 50).all(), (
            "Gaming RMSE values are unreasonably high"
        )

    def test_student_rmse_reasonable(self, student_metrics):
        assert (student_metrics["RMSE"] < 2.0).all(), (
            "Student RMSE values are unreasonably high "
            "(CGPA range is 0-4)"
        )

    def test_gaming_best_model_r2_above_threshold(
        self, gaming_metrics
    ):
        best_r2 = gaming_metrics["R2"].max()
        assert best_r2 > 0.85, (
            f"Best Gaming R2 is {best_r2:.4f}, "
            f"expected above 0.85"
        )

    def test_student_best_model_r2_above_threshold(
        self, student_metrics
    ):
        best_r2 = student_metrics["R2"].max()
        assert best_r2 > 0.90, (
            f"Best Student R2 is {best_r2:.4f}, "
            f"expected above 0.90"
        )

    def test_metrics_columns_complete(self, gaming_metrics):
        expected = ["RMSE", "MAE", "R2", "Adjusted R2"]
        for col in expected:
            assert col in gaming_metrics.columns, (
                f"Missing metric column: {col}"
            )


# =========================
# TEST GROUP 9: PROCESSED DATA INTEGRITY
# =========================

class TestProcessedDataIntegrity:
    """
    Verifies the integrity of all processed
    data files in data/processed/.
    """

    def test_processed_directory_has_6_files(self):
        if not os.path.exists(PROCESSED_DATA_DIR):
            pytest.skip("Processed directory does not exist yet")

        file_count = len([
            f for f in os.listdir(PROCESSED_DATA_DIR)
            if f.endswith(".csv")
        ])

        assert file_count == 6, (
            f"Expected 6 processed CSV files, found {file_count}"
        )

    def test_cleaned_gaming_shape(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_gaming.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Cleaned gaming file not found")

        df = pd.read_csv(path)
        assert df.shape[0] == 8000, (
            f"Expected 8000 rows, found {df.shape[0]}"
        )

    def test_cleaned_mental_shape(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_mental.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Cleaned mental file not found")

        df = pd.read_csv(path)
        assert df.shape[0] == 100, (
            f"Expected 100 rows, found {df.shape[0]}"
        )

    def test_cleaned_student_shape(self):
        path = os.path.join(
            PROCESSED_DATA_DIR,
            "cleaned_student.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Cleaned student file not found")

        df = pd.read_csv(path)
        assert df.shape[0] == 5000, (
            f"Expected 5000 rows, found {df.shape[0]}"
        )

    def test_feature_files_have_more_columns(self):
        cleaned = pd.read_csv(
            os.path.join(
                PROCESSED_DATA_DIR,
                "cleaned_gaming.csv"
            )
        )
        featured = pd.read_csv(
            os.path.join(
                PROCESSED_DATA_DIR,
                "final_gaming_features.csv"
            )
        )
        assert featured.shape[1] > cleaned.shape[1], (
            "Feature-enhanced dataset should have more columns "
            "than cleaned dataset"
        )


# =========================
# TEST GROUP 10: DEPENDENCY VERIFICATION
# =========================

class TestDependencies:
    """
    Verifies that all required Python packages
    can be imported without errors.
    """

    def test_import_pandas(self):
        import pandas
        assert pandas is not None

    def test_import_numpy(self):
        import numpy
        assert numpy is not None

    def test_import_sklearn(self):
        import sklearn
        assert sklearn is not None

    def test_import_matplotlib(self):
        import matplotlib
        assert matplotlib is not None

    def test_import_seaborn(self):
        import seaborn
        assert seaborn is not None

    def test_import_shap(self):
        import shap
        assert shap is not None

    def test_import_scipy(self):
        import scipy
        assert scipy is not None

    def test_import_joblib(self):
        import joblib
        assert joblib is not None
