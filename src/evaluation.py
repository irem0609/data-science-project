import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# SEABORN THEME
# =========================

sns.set_theme(style="whitegrid")

# =========================
# METRIC CALCULATION
# =========================

def calculate_metrics(y_true, y_pred, X_shape):
    """
    Calculates regression metrics.
    """

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    n, p = X_shape

    if n > p + 1:

        adj_r2 = 1 - (
            (1 - r2) * (n - 1) / (n - p - 1)
        )

    else:

        adj_r2 = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adjusted R2": adj_r2
    }

# =========================
# EVALUATE ALL MODELS
# =========================

def evaluate_all_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluates all trained models.
    """

    results = {}

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        results[name] = calculate_metrics(
            y_test,
            y_pred,
            X_test.shape
        )

    return results

# =========================
# SAVE METRICS + COMPARISON
# =========================

def save_and_plot_metrics(
    results_df,
    dataset_name,
    output_dir
):
    """
    1. Saves metrics as CSV
    2. Generates model comparison graph
    """

    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # SAVE CSV
    # =========================

    csv_path = os.path.join(
        output_dir,
        f"{dataset_name}_metrics.csv"
    )

    results_df.to_csv(
        csv_path,
        index=True
    )

    print(
        f"[SAVED] Metrics CSV saved: {csv_path}"
    )

    # =========================
    # PLOT
    # =========================

    fig, ax1 = plt.subplots(
        figsize=(10, 6)
    )

    # =========================
    # R2 BARPLOT
    # =========================

    sns.barplot(
        x=results_df.index,
        y='R2',
        data=results_df,
        ax=ax1,
        color='royalblue',
        alpha=0.7
    )

    ax1.set_ylabel(
        'R² Score',
        color='royalblue',
        fontweight='bold'
    )

    ax1.set_ylim(
        min(results_df['R2']) - 0.1,
        1.0
    )

    ax1.tick_params(
        axis='y',
        labelcolor='royalblue'
    )

    # =========================
    # RMSE LINEPLOT
    # =========================

    ax2 = ax1.twinx()

    sns.lineplot(
        x=results_df.index,
        y='RMSE',
        data=results_df,
        ax=ax2,
        color='darkred',
        marker='o',
        linewidth=2,
        markersize=8
    )

    ax2.set_ylabel(
        'RMSE',
        color='darkred',
        fontweight='bold'
    )

    ax2.tick_params(
        axis='y',
        labelcolor='darkred'
    )

    plt.title(
        f"{dataset_name} - Model Performance Comparison",
        fontweight='bold'
    )

    plt.xticks(rotation=45)

    fig.tight_layout()

    # =========================
    # SAVE FIGURE
    # =========================

    plot_path = os.path.join(
        output_dir,
        f"{dataset_name}_model_comparison.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"[SAVED] Comparison graph saved: {plot_path}"
    )

# =========================
# ACTUAL VS PREDICTED
# =========================

def plot_actual_vs_predicted(
    trained_models,
    X_test,
    y_test,
    dataset_name,
    output_dir
):
    """
    Generates Actual vs Predicted plots
    for all trained models.
    """

    os.makedirs(output_dir, exist_ok=True)

    num_models = len(trained_models)

    cols = 3

    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 5 * rows)
    )

    axes = axes.flatten()

    for i, (name, model) in enumerate(
        trained_models.items()
    ):

        y_pred = model.predict(X_test)

        ax = axes[i]

        # =========================
        # SCATTER PLOT
        # =========================

        sns.scatterplot(
            x=y_test,
            y=y_pred,
            ax=ax,
            alpha=0.6,
            color='teal'
        )

        # =========================
        # PERFECT PREDICTION LINE
        # =========================

        min_val = min(
            y_test.min(),
            y_pred.min()
        )

        max_val = max(
            y_test.max(),
            y_pred.max()
        )

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            lw=2,
            label='Perfect Prediction'
        )

        ax.set_title(
            name,
            fontweight='bold'
        )

        ax.set_xlabel(
            "Actual Values"
        )

        ax.set_ylabel(
            "Predicted Values"
        )

        ax.legend()

    # =========================
    # HIDE UNUSED AXES
    # =========================

    for j in range(i + 1, len(axes)):

        axes[j].axis('off')

    plt.suptitle(
        f"{dataset_name} - Actual vs Predicted",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )

    # =========================
    # SAVE FIGURE
    # =========================

    plot_path = os.path.join(
        output_dir,
        f"{dataset_name}_actual_vs_predicted.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"[SAVED] Actual vs Predicted plot saved: {plot_path}"
    )