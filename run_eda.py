"""
Standalone EDA script — runs all analyses and saves figures to outputs/figures/.
This does not require Jupyter. Just run: python run_eda.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Style ───
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

FIGURES_DIR = os.path.join('outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {name}")

# ═══════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════
print("\n=== 1. Data Loading ===")
student_df = pd.read_csv('data/processed/final_student_features.csv')
gaming_df  = pd.read_csv('data/processed/final_gaming_features.csv')
mental_df  = pd.read_csv('data/processed/final_mental_features.csv')

datasets = {'Student': student_df, 'Gaming': gaming_df, 'Mental': mental_df}
for name, df in datasets.items():
    print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")

EXCLUDE = ['student_id', 'source', 'is_female', 'is_male', 'is_other']

# ═══════════════════════════════════════════════════
# 2. HISTOGRAMS
# ═══════════════════════════════════════════════════
print("\n=== 2. Histograms ===")

for ds_name, df, cmap_name in [('student', student_df, 'viridis'),
                                 ('gaming', gaming_df, 'magma'),
                                 ('mental', mental_df, 'coolwarm')]:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]
    n_cols = 4
    n_rows = max(1, (len(num_cols) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    fig.suptitle(f'{ds_name.title()} Dataset - Numeric Variable Distributions',
                 fontsize=18, fontweight='bold', y=1.02)
    colors = sns.color_palette(cmap_name, len(num_cols))
    bins = 15 if ds_name == 'mental' else 30
    for i, col in enumerate(num_cols):
        ax = axes.flatten()[i]
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], edgecolor='white', alpha=0.7, bins=bins)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=8)
    for j in range(len(num_cols), n_rows * n_cols):
        axes.flatten()[j].set_visible(False)
    plt.tight_layout()
    save_fig(fig, f'histogram_{ds_name}.png')

# ═══════════════════════════════════════════════════
# 3. CORRELATION HEATMAPS
# ═══════════════════════════════════════════════════
print("\n=== 3. Correlation Heatmaps ===")

for ds_name, df, figsize in [('student', student_df, (14, 10)),
                                ('gaming', gaming_df, (18, 14)),
                                ('mental', mental_df, (12, 9))]:
    corr_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in EXCLUDE if c in df.columns], errors='ignore')
    fig, ax = plt.subplots(figsize=figsize)
    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fs = 8 if ds_name == 'gaming' else 9
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': fs}, vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
    ax.set_title(f'{ds_name.title()} Dataset - Correlation Matrix',
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_fig(fig, f'correlation_matrix_{ds_name}.png')

# ═══════════════════════════════════════════════════
# 4. BOXPLOTS
# ═══════════════════════════════════════════════════
print("\n=== 4. Boxplots ===")

boxplot_cols = {
    'student': ['age', 'attendance', 'study_hours_per_day', 'previous_cgpa',
                'sleep_hours_per_day', 'social_hours_week', 'final_cgpa',
                'study_efficiency', 'attendance_impact'],
    'gaming': ['age', 'gaming_hours', 'study_hours_per_day', 'sleep_hours_per_day',
               'attendance', 'device_usage', 'reaction_time_ms', 'addiction_score',
               'grades', 'final_cgpa', 'gaming_intensity', 'academic_pressure_index',
               'sleep_gaming_balance', 'total_screen_time'],
    'mental': ['age', 'year_of_study', 'marital_status', 'depression', 'anxiety',
               'panic_attack', 'seek_treatment', 'final_cgpa', 'total_mental_risk'],
}

palettes = {'student': 'Set2', 'gaming': 'husl', 'mental': 'Spectral'}

for ds_name, df in [('student', student_df), ('gaming', gaming_df), ('mental', mental_df)]:
    cols = [c for c in boxplot_cols[ds_name] if c in df.columns]
    n_box = len(cols)
    n_c = 5
    n_r = max(1, (n_box + n_c - 1) // n_c)
    fig, axes = plt.subplots(n_r, n_c, figsize=(24, n_r * 5))
    if n_r == 1:
        axes = np.array(axes).reshape(1, -1)
    fig.suptitle(f'{ds_name.title()} Dataset - Boxplots',
                 fontsize=18, fontweight='bold', y=1.02)
    colors = sns.color_palette(palettes[ds_name], n_box)
    for i, col in enumerate(cols):
        ax = axes.flatten()[i]
        sns.boxplot(y=df[col], ax=ax, color=colors[i], width=0.5,
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4})
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('')
    for j in range(n_box, n_r * n_c):
        axes.flatten()[j].set_visible(False)
    plt.tight_layout()
    save_fig(fig, f'boxplot_{ds_name}.png')

# ═══════════════════════════════════════════════════
# 5. SCATTERPLOTS
# ═══════════════════════════════════════════════════
print("\n=== 5. Scatterplots ===")

scatter_config = {
    'student': {
        'pairs': [('study_hours_per_day', 'final_cgpa'), ('attendance', 'final_cgpa'),
                  ('previous_cgpa', 'final_cgpa'), ('sleep_hours_per_day', 'final_cgpa'),
                  ('social_hours_week', 'final_cgpa'), ('study_efficiency', 'final_cgpa'),
                  ('attendance_impact', 'final_cgpa'), ('age', 'final_cgpa')],
        'color': '#3498db',
    },
    'gaming': {
        'pairs': [('gaming_hours', 'final_cgpa'), ('study_hours_per_day', 'final_cgpa'),
                  ('sleep_hours_per_day', 'final_cgpa'), ('attendance', 'final_cgpa'),
                  ('addiction_score', 'final_cgpa'), ('reaction_time_ms', 'final_cgpa'),
                  ('gaming_intensity', 'final_cgpa'), ('academic_pressure_index', 'final_cgpa'),
                  ('sleep_gaming_balance', 'final_cgpa'), ('total_screen_time', 'final_cgpa'),
                  ('device_usage', 'final_cgpa'), ('social_hours_day', 'final_cgpa')],
        'color': '#9b59b6',
    },
    'mental': {
        'pairs': [('depression', 'final_cgpa'), ('anxiety', 'final_cgpa'),
                  ('panic_attack', 'final_cgpa'), ('total_mental_risk', 'final_cgpa'),
                  ('age', 'final_cgpa'), ('year_of_study', 'final_cgpa'),
                  ('marital_status', 'final_cgpa'), ('seek_treatment', 'final_cgpa')],
        'color': '#e67e22',
    },
}

for ds_name, df in [('student', student_df), ('gaming', gaming_df), ('mental', mental_df)]:
    cfg = scatter_config[ds_name]
    pairs = [(x, y) for x, y in cfg['pairs'] if x in df.columns and y in df.columns]
    n_p = len(pairs)
    n_c = 4
    n_r = max(1, (n_p + n_c - 1) // n_c)
    alpha = 0.5 if ds_name == 'mental' else 0.2
    s = 30 if ds_name == 'mental' else 12

    fig, axes = plt.subplots(n_r, n_c, figsize=(22, n_r * 5))
    if n_r == 1:
        axes = np.array(axes).reshape(1, -1)
    fig.suptitle(f'{ds_name.title()} - Feature Relationships with final_cgpa',
                 fontsize=18, fontweight='bold', y=1.02)
    for i, (x_col, y_col) in enumerate(pairs):
        ax = axes.flatten()[i]
        sns.regplot(x=df[x_col], y=df[y_col], ax=ax,
                    scatter_kws={'alpha': alpha, 's': s, 'color': cfg['color']},
                    line_kws={'color': '#e74c3c', 'linewidth': 2})
        corr_val = df[[x_col, y_col]].corr().iloc[0, 1]
        ax.set_title(f'{x_col}\n(r = {corr_val:.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
    for j in range(n_p, n_r * n_c):
        axes.flatten()[j].set_visible(False)
    plt.tight_layout()
    save_fig(fig, f'scatterplot_{ds_name}_cgpa.png')

# ═══════════════════════════════════════════════════
# 6. PAIRPLOTS
# ═══════════════════════════════════════════════════
print("\n=== 6. Pairplot Analysis ===")

pairplot_config = {
    'student': {
        'cols': ['study_hours_per_day', 'attendance', 'previous_cgpa',
                 'sleep_hours_per_day', 'social_hours_week', 'final_cgpa'],
        'color': '#2ecc71', 'kde_color': '#27ae60',
    },
    'gaming': {
        'cols': ['gaming_hours', 'study_hours_per_day', 'sleep_hours_per_day',
                 'attendance', 'addiction_score', 'final_cgpa'],
        'color': '#8e44ad', 'kde_color': '#9b59b6',
    },
    'mental': {
        'cols': ['final_cgpa', 'depression', 'anxiety', 'panic_attack',
                 'total_mental_risk', 'age'],
        'color': '#e74c3c', 'kde_color': '#c0392b',
    },
}

for ds_name, df in [('student', student_df), ('gaming', gaming_df), ('mental', mental_df)]:
    cfg = pairplot_config[ds_name]
    cols = [c for c in cfg['cols'] if c in df.columns]
    alpha = 0.5 if ds_name == 'mental' else 0.3
    s = 30 if ds_name == 'mental' else 12
    g = sns.pairplot(df[cols], diag_kind='kde',
                     plot_kws={'alpha': alpha, 's': s, 'color': cfg['color']},
                     diag_kws={'color': cfg['kde_color'], 'fill': True, 'alpha': 0.5})
    g.figure.suptitle(f'{ds_name.title()} Dataset - Pairplot Analysis',
                      fontsize=16, fontweight='bold', y=1.02)
    save_fig(g.figure, f'pairplot_{ds_name}.png')

# ═══════════════════════════════════════════════════
# 7. CATEGORICAL BOXPLOTS
# ═══════════════════════════════════════════════════
print("\n=== 7. Categorical Boxplots ===")

# Gaming: Stress Level & Gender vs final_cgpa
if 'stress_level' in gaming_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.boxplot(x='stress_level', y='final_cgpa', data=gaming_df, ax=axes[0], palette='viridis', width=0.6)
    axes[0].set_title('Final CGPA by Stress Level (Gaming)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Stress Level (Encoded)', fontsize=12)
    axes[0].set_ylabel('Final CGPA', fontsize=12)
    if 'gender' in gaming_df.columns:
        sns.boxplot(x='gender', y='final_cgpa', data=gaming_df, ax=axes[1], palette='Set2', width=0.6)
        axes[1].set_title('Final CGPA by Gender (Gaming)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Gender (0=F, 1=M, 2=Other)', fontsize=12)
        axes[1].set_ylabel('Final CGPA', fontsize=12)
    plt.tight_layout()
    save_fig(fig, 'boxplot_gaming_categorical_vs_cgpa.png')

# Student: Gender & Major vs final_cgpa
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
if 'gender' in student_df.columns:
    sns.boxplot(x='gender', y='final_cgpa', data=student_df, ax=axes[0], palette='coolwarm', width=0.5)
    axes[0].set_title('Final CGPA by Gender (Student)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gender (0=F, 1=M)', fontsize=12)
    axes[0].set_ylabel('Final CGPA', fontsize=12)
if 'major' in student_df.columns:
    sns.boxplot(x='major', y='final_cgpa', data=student_df, ax=axes[1], palette='Set3', width=0.6)
    axes[1].set_title('Final CGPA by Major (Student)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Major (Encoded)', fontsize=12)
    axes[1].set_ylabel('Final CGPA', fontsize=12)
plt.tight_layout()
save_fig(fig, 'boxplot_student_categorical_vs_cgpa.png')

# Mental: Depression, Anxiety, Total Mental Risk vs final_cgpa
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
if 'depression' in mental_df.columns:
    sns.boxplot(x='depression', y='final_cgpa', data=mental_df, ax=axes[0], palette='RdYlBu_r', width=0.5)
    axes[0].set_title('Final CGPA by Depression', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Depression (0=No, 1=Yes)', fontsize=12)
    axes[0].set_ylabel('Final CGPA', fontsize=12)
if 'anxiety' in mental_df.columns:
    sns.boxplot(x='anxiety', y='final_cgpa', data=mental_df, ax=axes[1], palette='YlOrRd', width=0.5)
    axes[1].set_title('Final CGPA by Anxiety', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Anxiety (0=No, 1=Yes)', fontsize=12)
    axes[1].set_ylabel('Final CGPA', fontsize=12)
if 'total_mental_risk' in mental_df.columns:
    sns.boxplot(x='total_mental_risk', y='final_cgpa', data=mental_df, ax=axes[2], palette='Spectral', width=0.5)
    axes[2].set_title('Final CGPA by Total Mental Risk Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Total Mental Risk (0-3)', fontsize=12)
    axes[2].set_ylabel('Final CGPA', fontsize=12)
plt.tight_layout()
save_fig(fig, 'boxplot_mental_categorical_vs_cgpa.png')

# ═══════════════════════════════════════════════════
# 8. CORRELATION RANKINGS
# ═══════════════════════════════════════════════════
print("\n=== 8. Correlation Summaries ===")

def top_correlations(df, target, name, n=15):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(columns=[c for c in EXCLUDE if c in numeric_df.columns], errors='ignore')
    if target not in numeric_df.columns:
        print(f"  [!] {target} not found.")
        return None
    corr = numeric_df.corr()[target].drop(target)
    corr_abs = corr.abs().sort_values(ascending=False)
    print(f"\n  --- {name} : Strongest Correlations with {target} ---")
    for col in corr_abs.head(n).index:
        val = corr[col]
        arrow = "Positive" if val > 0 else "Negative"
        print(f"    {col:<35} r = {val:+.4f}  ({arrow})")
    return corr

corr_student = top_correlations(student_df, 'final_cgpa', 'Student')
corr_gaming  = top_correlations(gaming_df,  'final_cgpa', 'Gaming')
corr_mental  = top_correlations(mental_df,  'final_cgpa', 'Mental')

# ═══════════════════════════════════════════════════
# 9. SUMMARY BAR CHART
# ═══════════════════════════════════════════════════
print("\n=== 9. Summary Visualization ===")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Correlations with Target Variable (final_cgpa) - Summary',
             fontsize=18, fontweight='bold', y=1.02)

for ax, (ds_name, df) in zip(axes, [('Student', student_df), ('Gaming', gaming_df), ('Mental', mental_df)]):
    corr_data = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in EXCLUDE if c in df.columns], errors='ignore'
    ).corr()['final_cgpa'].drop('final_cgpa').sort_values()
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr_data.values]
    corr_data.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title(ds_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient (r)')
    ax.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
save_fig(fig, 'summary_target_correlations.png')

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  COMPLETED!")
print("=" * 60)

# List all saved figures
figs = sorted(os.listdir(FIGURES_DIR))
print(f"\n  Total {len(figs)} figures created:")
for f in figs:
    size_kb = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
    print(f"    - {f} ({size_kb:.0f} KB)")
