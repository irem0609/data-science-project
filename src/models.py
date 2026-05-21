from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# MODEL PIPELINES
# =========================

def get_model_pipelines():
    """
    Returns pipelines that include:

    - StandardScaler
    - Machine Learning model

    Scaling is applied only to the training data
    in order to prevent data leakage.
    """

    pipelines = {

        "Linear Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),

        "Decision Tree": Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeRegressor(
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                random_state=42
            ))
        ]),

        "SVR": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR())
        ]),

        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor())
        ]),

        "Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                random_state=42
            ))
        ])
    }

    # =========================
    # HYPERPARAMETER GRIDS
    # =========================

    param_grids = {

        # Linear Regression
        "Linear Regression": {},

        # Decision Tree
        "Decision Tree": {
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        },

        # Random Forest
        "Random Forest": {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        },

        # SVR
        "SVR": {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        },

        # KNN
        "KNN": {
            'model__n_neighbors': [3, 5, 7, 10],
            'model__weights': ['uniform', 'distance']
        },

        # Gradient Boosting
        "Gradient Boosting": {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5]
        }
    }

    return pipelines, param_grids

# =========================
# TRAIN MODELS WITH CV
# =========================

def train_models_with_cv(
    X_train,
    y_train
):
    """
    Trains all models using:

    - Pipeline
    - Cross Validation
    - GridSearchCV

    Returns the best tuned models.
    """

    pipelines, param_grids = get_model_pipelines()

    best_models = {}

    best_scores = {}

    for name, pipeline in pipelines.items():

        print(
            f"\n[{name}] "
            f"Hyperparameter tuning and "
            f"Cross Validation started..."
        )

        # =========================
        # GRID SEARCH
        # =========================

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # =========================
        # TRAIN MODEL
        # =========================

        grid_search.fit(
            X_train,
            y_train
        )

        # =========================
        # RESULTS
        # =========================

        print(
            f"   => Best Parameters: "
            f"{grid_search.best_params_}"
        )

        print(
            f"   => Best CV R2 Score: "
            f"{grid_search.best_score_:.4f}"
        )

        # Best tuned model
        best_models[name] = (
            grid_search.best_estimator_
        )

        # Best CV score
        best_scores[name] = (
            grid_search.best_score_
        )

    # =========================
    # FINAL SUMMARY
    # =========================

    print("\n=========================")
    print("BEST CV SCORES")
    print("=========================")

    for model_name, score in best_scores.items():

        print(
            f"{model_name}: "
            f"{score:.4f}"
        )

    return best_models