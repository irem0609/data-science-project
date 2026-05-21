# Setting up the development environment

First cd into the project location:

```bash
cd data-science-project
```

It is recommended to use a virtual environment to manage dependencies for this project. You can create a virtual environment using the following command:
On Linux:

```bash
python3 -m venv .venv
```

On Windows:

```powershell
py -m venv .venv
```

Then activate the virtual environment

On Linux:

```bash
source .venv/bin/activate
```

On Windows:

- Cmd:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- PowerShell:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Finally, install the dependencies:

```bash
pip install -r requirements.txt
```

This will install the required package for the project.

# Running the data cleaning script

Then you can run the data cleaning script with:

```bash
python src/data_cleaning.py
```

This script needs to be run at the root of the project (the same level as the `src` folder) to work properly. It will read raw data from `data/raw/` directory, clean it, and save the cleaned data to `data/processed/` directory.

# Running the data enrichment script

After cleaning the data, the next step is to generate the analytical attributes required for the machine learning models. You can run the feature engineering pipeline with:

```bash
python src/features.py
```

This script reads the cleaned datasets from the `data/processed/` directory, calculates new scientific metrics (such as the academic pressure index, sleep-to-game ratio, and total mental risk), and saves the enriched datasets back to the `data/processed/` directory with a final\_ prefix. These final files are ready for model training.

# Running the Machine Learning Pipeline

After generating the final feature-engineered datasets, you can run the machine learning pipeline with:

```bash
python src/main.py
```

This pipeline automatically performs:

- Train/Test Split
- Data Scaling
- Cross Validation
- Hyperparameter Optimization
- Model Training
- Performance Evaluation
- Explainability Analysis
- Visualization Generation

---

# Machine Learning Models

The following regression models are implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Regressor

---

# Evaluation Metrics

The project evaluates model performance using:

- RMSE
- MAE
- R² Score
- Adjusted R²

---

# Explainability Methods

To interpret model behavior, the project includes:

- Feature Importance
- Permutation Importance
- SHAP Summary Analysis

---

# Generated Reports

All outputs are automatically saved into:

```text
reports/
```

Generated outputs include:

- Metrics CSV files
- Model comparison graphs
- Actual vs Predicted plots
- Feature importance graphs
- Permutation importance graphs
- SHAP summary plots

---

# Important Notes

- Each dataset is modeled independently.
- Data leakage prevention techniques are applied.
- StandardScaler is integrated into ML pipelines.
- Hyperparameter optimization is performed using GridSearchCV.
- Cross-validation uses 5-Fold CV.

# Project Structure

```text
data-science-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│
├── src/
│   ├── data_cleaning.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── main.py
│
├── requirements.txt
└── README.md
```
