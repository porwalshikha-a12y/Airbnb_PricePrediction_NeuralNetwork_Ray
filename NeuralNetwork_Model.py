"""
NeuralNetwork_Model.py — MLP regression baseline using sklearn's default loky
multiprocessing backend (no Ray).

Algorithm:    sklearn.neural_network.MLPRegressor (with early stopping)
Parallelism:  sklearn / joblib loky backend (n_jobs=-1)

Public entry point: run_nn(df) -> dict matching the project's contract:
    {best_params, model, metrics_log, metrics_real,
     elapsed_seconds, backend}
"""

import warnings
warnings.filterwarnings("ignore")

from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Feature engineering (pandas)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    p_lo = df["price"].quantile(0.05)
    p_hi = df["price"].quantile(0.95)
    df["price"] = df["price"].clip(p_lo, p_hi)
    df["price_log"] = np.log1p(df["price"])

    if {"latitude", "longitude"}.issubset(df.columns):
        df["distance_to_center"] = np.sqrt(
            (df["latitude"] - df["latitude"].mean()) ** 2
            + (df["longitude"] - df["longitude"].mean()) ** 2
        )

    if "amenities" in df.columns:
        df["amenities_count"] = (
            df["amenities"].astype(str).apply(lambda x: len(x.split(",")))
        )
        df = df.drop(columns=["amenities"])

    if {"bedrooms", "bathrooms"}.issubset(df.columns):
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.1)
    if {"accommodates", "bedrooms"}.issubset(df.columns):
        df["bedrooms_per_person"] = df["bedrooms"] / (df["accommodates"] + 0.1)

    review_cols = [c for c in df.columns if "review_scores" in c.lower()]
    if len(review_cols) > 1:
        df["avg_review_score"] = df[review_cols].mean(axis=1)

    if "host_since" in df.columns:
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_years"] = (pd.Timestamp.now() - df["host_since"]).dt.days / 365
        df = df.drop(columns=["host_since"])

    for col in ["instant_bookable", "host_is_superhost", "host_identity_verified"]:
        if col in df.columns:
            df[col] = (
                df[col].map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0)
            )

    return df


# ---------------------------------------------------------------------------
# 2. Encoding (robust against pyarrow string dtype)
# ---------------------------------------------------------------------------
DROP_TEXT = {"name", "description", "amenities", "neighbourhood_cleansed", "id"}


def encode_features(X_train, X_test):
    drop = [c for c in DROP_TEXT if c in X_train.columns]
    if drop:
        X_train = X_train.drop(columns=drop)
        X_test  = X_test.drop(columns=drop)

    for c in ["host_response_rate", "host_acceptance_rate"]:
        if c in X_train.columns:
            X_train[c] = (
                pd.to_numeric(
                    X_train[c].astype(str).str.rstrip("%"), errors="coerce"
                ).fillna(0) / 100.0
            )
            X_test[c] = (
                pd.to_numeric(
                    X_test[c].astype(str).str.rstrip("%"), errors="coerce"
                ).fillna(0) / 100.0
            )

    cat_cols = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].astype("object")
        X_test[c]  = X_test[c].astype("object")

    high_card_cols = [c for c in cat_cols if X_train[c].nunique() > 20]
    low_card_cols  = [c for c in cat_cols if X_train[c].nunique() <= 20]

    for col in high_card_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_map).fillna(0).astype(float)
        X_test[col]  = X_test[col].map(freq_map).fillna(0).astype(float)

    X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=low_card_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    leftover = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if leftover:
        print(f"WARNING: dropping unencodable columns: {leftover}")
        X_train = X_train.drop(columns=leftover)
        X_test  = X_test.drop(columns=leftover)

    return (
        X_train.fillna(0).values.astype(float),
        X_test.fillna(0).values.astype(float),
    )


# ---------------------------------------------------------------------------
# 3. Public entry point
# ---------------------------------------------------------------------------
def run_nn(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_splits: int = 5,
):
    """
    Multi-Layer Perceptron regression with sklearn-native (loky) parallelism
    for hyperparameter search.
    """
    print("\n=== Neural Network (sklearn + loky backend) STARTED ===")
    t_start = perf_counter()

    df = engineer_features(df)

    y = df["price_log"]
    X = df.drop(columns=["price", "price_log"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_np, X_test_np = encode_features(X_train, X_test)
    y_train_np = y_train.values
    y_test_np  = y_test.values

    print(f"Train shape: {X_train_np.shape} | Test shape: {X_test_np.shape}")

    # Pipeline + grid (mirrors NeuralNetworks_Ray.py's search space)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "mlp__activation":         ["relu", "tanh"],
        "mlp__alpha":              [0.0001, 0.001],
        "mlp__learning_rate_init": [0.001, 0.005],
        "mlp__batch_size":         [32, 64],
    }
    cv = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)

    n_combos = (
        len(param_grid["mlp__hidden_layer_sizes"])
        * len(param_grid["mlp__activation"])
        * len(param_grid["mlp__alpha"])
        * len(param_grid["mlp__learning_rate_init"])
        * len(param_grid["mlp__batch_size"])
    )
    print(
        f"\nRunning GridSearchCV (sklearn / loky backend): "
        f"{n_combos} combos × {n_cv_splits} folds = {n_combos * n_cv_splits} fits"
    )

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs.fit(X_train_np, y_train_np)

    best_model = gs.best_estimator_
    best_params = {
        k.replace("mlp__", ""): v for k, v in gs.best_params_.items()
    }
    print(f"\nBest Params: {best_params}")
    print(f"Best CV RMSE: {-gs.best_score_:.4f}")

    # Test-set evaluation
    y_pred = best_model.predict(X_test_np)

    metrics_log = {
        "rmse": float(np.sqrt(mean_squared_error(y_test_np, y_pred))),
        "mae":  float(mean_absolute_error(y_test_np, y_pred)),
        "r2":   float(r2_score(y_test_np, y_pred)),
    }
    metrics_real = {
        "rmse": float(np.sqrt(mean_squared_error(
            np.expm1(y_test_np), np.expm1(y_pred)
        ))),
        "mae":  float(mean_absolute_error(
            np.expm1(y_test_np), np.expm1(y_pred)
        )),
        "r2": float(r2_score(
            np.expm1(y_test_np), np.expm1(y_pred)
        ))
    }

    elapsed = perf_counter() - t_start

    print("\nNeural Network Test Performance (Log Scale):")
    print(
        f"  RMSE: {metrics_log['rmse']:.4f} | "
        f"MAE: {metrics_log['mae']:.4f} | "
        f"R²: {metrics_log['r2']:.4f}"
    )
    print("\nNeural Network Test Performance (Real Price Scale):")
    print(
        f"  RMSE: {metrics_real['rmse']:.2f} | "
        f"MAE: {metrics_real['mae']:.2f} |"
        f"R²: {metrics_real['r2']:.2f}"
    )
    print(f"\nTotal wall-clock: {elapsed:.1f}s  (backend: loky)")
    print("=== Neural Network (sklearn + loky backend) FINISHED ===")

    return {
        "best_params":     best_params,
        "model":           best_model,
        "metrics_log":     metrics_log,
        "metrics_real":    metrics_real,
        "elapsed_seconds": float(elapsed),
        "backend":         "loky",
    }


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    run_nn(df)
