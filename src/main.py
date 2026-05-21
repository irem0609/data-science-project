import os
import pandas as pd

from sklearn.model_selection import train_test_split

from models import train_models_with_cv

from evaluation import (
    evaluate_all_models,
    save_and_plot_metrics,
    plot_actual_vs_predicted
)

from explainability import (
    plot_feature_importance,
    calc_permutation_importance,
    apply_shap
)

# =========================
# DATA PREPARATION
# =========================

def prepare_data(df, target_col):
    """
    Prepares the dataset for Machine Learning.

    1. Separates the target column
    2. Removes columns that may cause data leakage
    3. Applies One-Hot Encoding to categorical columns
    """

    all_possible_targets = [
        'final_cgpa',
        'grades'
    ]

    base_drop_cols = [
        'student_id',
        'source',
        'cgpa_range'
    ]

     # Drop other target columns except the selected target
    drop_cols = base_drop_cols + all_possible_targets

    cols_to_drop = [
        col for col in drop_cols
        if col in df.columns
    ]

    # Separate features and target
    X = df.drop(columns=cols_to_drop)

    y = df[target_col]

    # =========================
    # ONE HOT ENCODING
    # =========================

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y

# =========================
# SINGLE DATASET PIPELINE
# =========================

def process_single_dataset(
    file_path,
    dataset_name,
    target_col,
    reports_dir
):

    print(f"\n\n{'#'*60}")

    print(
        f"STARTING: "
        f"{dataset_name.upper()} "
        f"(Target: {target_col})"
    )

    print(f"{'#'*60}")

    # =========================
    # FILE CHECK
    # =========================

    if not os.path.exists(file_path):

        print(
            f"[ERROR] "
            f"{file_path} not found!"
        )

        return

    # =========================
    # LOAD DATA
    # =========================

    df = pd.read_csv(file_path)

    # =========================
    # TARGET CHECK
    # =========================

    if target_col not in df.columns:

        print(
            f"[ERROR] "
            f"Target variable "
            f"'{target_col}' not found!"
        )

        return

    # =========================
    # PREPARE DATA
    # =========================

    X, y = prepare_data(
        df,
        target_col
    )

    # =========================
    # TRAIN TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n--- DATASET INFO ---")

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"Feature count: "
        f"{X.shape[1]}"
    )

    # =========================
    # MODEL TRAINING
    # =========================

    trained_pipelines = train_models_with_cv(
        X_train,
        y_train
    )

    # =========================
    # MODEL EVALUATION
    # =========================

    results = evaluate_all_models(
        trained_pipelines,
        X_test,
        y_test
    )

    results_df = pd.DataFrame(results).T

    print("\n--- MODEL PERFORMANCE ---")

    print(
        results_df.round(4)
    )

    # =========================
    # SAVE ALL MODEL RESULTS
    # =========================

    results_csv_path = os.path.join(
        reports_dir,
        f"{dataset_name}_all_model_results.csv"
    )

    results_df.to_csv(
        results_csv_path
    )

    print(
        f"[SAVED] "
        f"{results_csv_path}"
    )

    # =========================
    # METRICS + VISUALIZATION
    # =========================

    save_and_plot_metrics(
        results_df,
        dataset_name,
        reports_dir
    )

    plot_actual_vs_predicted(
        trained_pipelines,
        X_test,
        y_test,
        dataset_name,
        reports_dir
    )

    # =========================
    # BEST MODEL SELECTION
    # =========================

    best_model_name = (
        results_df['R2']
        .idxmax()
    )

    best_pipeline = (
        trained_pipelines[
            best_model_name
        ]
    )

    print(
        f"\n>> BEST MODEL: "
        f"{best_model_name} "
        f"(R2: "
        f"{results_df.loc[best_model_name, 'R2']:.4f})"
    )

    # =========================
    # EXPLAINABILITY PREPARATION
    # =========================

    actual_model = (
        best_pipeline
        .named_steps['model']
    )

    X_train_scaled = (
        best_pipeline
        .named_steps['scaler']
        .transform(X_train)
    )

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    if best_model_name in [
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting"
    ]:

        plot_feature_importance(
            actual_model,
            X.columns,
            model_name=(
                f"{dataset_name} - "
                f"{best_model_name}"
            )
        )

    else:

        calc_permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            model_name=(
                f"{dataset_name} - "
                f"{best_model_name}"
            )
        )

    # =========================
    # SHAP
    # =========================

    apply_shap(
        actual_model,
        X_train_scaled_df,
        model_name=(
            f"{dataset_name} - "
            f"{best_model_name}"
        )
    )

# =========================
# MAIN
# =========================

def main():

    # Source directory
    src_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Main project directory
    proj_root = os.path.dirname(
        src_dir
    )

    # =========================
    # PATHS
    # =========================

    processed_path = os.path.join(
        proj_root,
        "data",
        "processed"
    )

    reports_dir = os.path.join(
        proj_root,
        "reports"
    )

    # Ensure reports directory exists
    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    # =========================
    # DATASET CONFIG
    # =========================

    datasets_config = {

        "Gaming": {
            "path": os.path.join(
                processed_path,
                "final_gaming_features.csv"
            ),
            "target": "grades"
        },

        "Mental": {
            "path": os.path.join(
                processed_path,
                "final_mental_features.csv"
            ),
            "target": "final_cgpa"
        },

        "Student": {
            "path": os.path.join(
                processed_path,
                "final_student_features.csv"
            ),
            "target": "final_cgpa"
        }
    }

    # =========================
    # RUN ALL DATASETS
    # =========================

    for name, config in datasets_config.items():

        process_single_dataset(
            file_path=config["path"],
            dataset_name=name,
            target_col=config["target"],
            reports_dir=reports_dir
        )

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    main()