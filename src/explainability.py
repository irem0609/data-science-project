import os
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

# =========================
# OUTPUT DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

# =========================
# FEATURE IMPORTANCE
# =========================

def plot_feature_importance(
    model,
    feature_names,
    model_name
):
    """
    Generates built-in feature importance
    for tree-based models.
    """

    safe_model_name = (
        model_name
        .replace(" ", "_")
        .lower()
    )

    if hasattr(model, 'feature_importances_'):

        importances = model.feature_importances_

        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Select top 10 most important features
        df_imp = (
            df_imp
            .sort_values(
                by='Importance',
                ascending=False
            )
            .head(10)
        )

        # =========================
        # SAVE CSV
        # =========================

        csv_save_path = os.path.join(
            REPORT_DIR,
            f"{safe_model_name}_feature_importance.csv"
        )

        df_imp.to_csv(
            csv_save_path,
            index=False
        )

        print(
            f"[SAVED] "
            f"{csv_save_path}"
        )

        # =========================
        # PLOT
        # =========================

        plt.figure(figsize=(10, 6))

        plt.barh(
            df_imp['Feature'][::-1],
            df_imp['Importance'][::-1],
            color='skyblue'
        )

        plt.title(
            f"{model_name} - "
            f"Top 10 Feature Importance"
        )

        plt.xlabel(
            "Importance Score"
        )

        plt.tight_layout()

        save_path = os.path.join(
            REPORT_DIR,
            f"{safe_model_name}_feature_importance.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(
            f"[SAVED] "
            f"{save_path}"
        )

# =========================
# PERMUTATION IMPORTANCE
# =========================

def calc_permutation_importance(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Calculates permutation importance
    for KNN, SVR, and Linear models.
    """

    safe_model_name = (
        model_name
        .replace(" ", "_")
        .lower()
    )

    print(
        f"\n--- Calculating "
        f"Permutation Importance "
        f"for {model_name} ---"
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    df_perm = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean
    })

    # Select top 10 most important features
    df_perm = (
        df_perm
        .sort_values(
            by='Importance',
            ascending=False
        )
        .head(10)
    )

    print(df_perm)

    # =========================
    # SAVE CSV
    # =========================

    csv_save_path = os.path.join(
        REPORT_DIR,
        f"{safe_model_name}_permutation_importance.csv"
    )

    df_perm.to_csv(
        csv_save_path,
        index=False
    )

    print(
        f"[SAVED] "
        f"{csv_save_path}"
    )

    # =========================
    # PLOT
    # =========================

    plt.figure(figsize=(10, 6))

    plt.barh(
        df_perm['Feature'][::-1],
        df_perm['Importance'][::-1],
        color='orange'
    )

    plt.title(
        f"{model_name} - "
        f"Permutation Importance"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.tight_layout()

    save_path = os.path.join(
        REPORT_DIR,
        f"{safe_model_name}_permutation_importance.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"[SAVED] "
        f"{save_path}"
    )

# =========================
# SHAP
# =========================

def apply_shap(
    model,
    X_train,
    model_name
):
    """
    Generates SHAP summary plots
    for explainability analysis.
    """

    print(
        f"\n[{model_name}] "
        f"Calculating SHAP values..."
    )

    safe_model_name = (
        model_name
        .replace(" ", "_")
        .lower()
    )

    try:

        # =========================
        # SAMPLE LIMIT
        # =========================

        if len(X_train) > 300:

            X_train = X_train.sample(
                300,
                random_state=42
            )

        # =========================
        # TREE MODELS
        # =========================

        if (
            "Random Forest" in model_name
            or "Decision Tree" in model_name
            or "Gradient Boosting" in model_name
        ):

            explainer = shap.TreeExplainer(
                model
            )

            shap_values = explainer.shap_values(
                X_train
            )

        # =========================
        # LINEAR MODELS
        # =========================

        elif "Linear Regression" in model_name:

            explainer = shap.LinearExplainer(
                model,
                X_train
            )

            shap_values = explainer.shap_values(
                X_train
            )

        # =========================
        # OTHER MODELS
        # =========================

        else:

            print(
                f"{model_name} was skipped "
                f"because KernelExplainer "
                f"can be extremely slow."
            )

            return

        # =========================
        # SHAP SUMMARY PLOT
        # =========================

        plt.figure(figsize=(12, 7))

        shap.summary_plot(
            shap_values,
            X_train,
            show=False
        )

        plt.title(
            f"{model_name} - "
            f"SHAP Summary Plot"
        )

        plt.tight_layout()

        save_path = os.path.join(
            REPORT_DIR,
            f"{safe_model_name}_shap_summary.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(
            f"[SAVED] "
            f"{save_path}"
        )

    except Exception as e:

        print(
            f"[ERROR] "
            f"Failed to generate SHAP plot: {e}"
        )