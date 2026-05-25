"""
# =========================
# RUN ALL - END TO END PIPELINE EXECUTOR
# =========================

Executes the entire data science pipeline in the correct order:
    1. Data Loading and Cleaning  (load_and_clean.py)
    2. Feature Engineering        (features.py)
    3. Model Training, Evaluation, and Explainability (main.py)

Usage:
    python qa_debug_and_test_outputs/run_all.py
"""

import subprocess
import sys
import os
import time

# =========================
# PATH CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# =========================
# PIPELINE STEPS
# =========================

PIPELINE_STEPS = [
    {
        "name": "Step 1: Data Loading and Cleaning",
        "script": os.path.join(SRC_DIR, "load_and_clean.py"),
        "description": "Loads raw CSV files, handles missing values, "
                       "standardizes column names, and saves cleaned data."
    },
    {
        "name": "Step 2: Feature Engineering",
        "script": os.path.join(SRC_DIR, "features.py"),
        "description": "Generates engineered features from cleaned datasets "
                       "and saves final feature files."
    },
    {
        "name": "Step 3: Model Training, Evaluation, and Explainability",
        "script": os.path.join(SRC_DIR, "main.py"),
        "description": "Trains 6 regression models with GridSearchCV, "
                       "evaluates performance, generates comparison plots, "
                       "feature importance, and SHAP analysis."
    }
]


# =========================
# STEP EXECUTOR
# =========================

def run_step(step_info):
    """
    Executes a single pipeline step and returns
    whether it passed or failed.
    """

    print(f"\n{'='*60}")
    print(f"  {step_info['name']}")
    print(f"  {step_info['description']}")
    print(f"{'='*60}")

    script_path = step_info["script"]

    if not os.path.exists(script_path):
        print(f"  [FAIL] Script not found: {script_path}")
        return False

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start_time

        if result.stdout:
            print(result.stdout)

        if result.returncode == 0:
            print(f"  [PASS] Completed in {elapsed:.2f} seconds.")
            return True
        else:
            print(f"  [FAIL] Exit code: {result.returncode}")
            if result.stderr:
                print(f"  Error details:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return False


# =========================
# MAIN
# =========================

def main():

    print("\n" + "#" * 60)
    print("#  END-TO-END PIPELINE EXECUTION")
    print("#" * 60)

    total_start = time.time()
    results = []

    for step in PIPELINE_STEPS:
        passed = run_step(step)
        results.append((step["name"], passed))

        if not passed:
            print(f"\n  [ABORT] Pipeline stopped at: {step['name']}")
            break

    # =========================
    # SUMMARY
    # =========================

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("  PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(PIPELINE_STEPS)

    print(f"\n  Result: {passed_count}/{total_count} steps passed.")
    print(f"  Total execution time: {total_elapsed:.2f} seconds.")

    if passed_count == total_count:
        print("  Pipeline completed successfully.\n")
    else:
        print("  Pipeline failed. Check error details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
