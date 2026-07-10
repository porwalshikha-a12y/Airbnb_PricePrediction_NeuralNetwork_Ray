import pandas as pd
import numpy as np
from time import perf_counter
import os

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


# -------------------------------
# 1. LOAD DATA
# -------------------------------
file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data/Cleansed/airbnb_model_input.csv"
)

df = pd.read_csv(file_path)

# -------------------------------
# 2. PREPROCESS
# -------------------------------
df["price_log"] = np.log1p(df["price"])

X = df.drop(columns=["price", "price_log"])
y = df["price_log"]

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Convert to NumPy (IMPORTANT for Ray)
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)


# -------------------------------
# 3. RAY + ASHA
# -------------------------------
print("\n=== RAY + ASHA ===")

os.environ["RAY_memory_usage_threshold"] = "0.95"

ray.init(ignore_reinit_error=True)

# Put data into Ray memory
X_ref = ray.put(X_train_np)
y_ref = ray.put(y_train_np)


def train_model(config):
    X = ray.get(X_ref)
    y = ray.get(y_ref)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation=config["activation"],
            alpha=config["alpha"],
            learning_rate_init=config["learning_rate_init"],
            batch_size=config["batch_size"],
            max_iter=200,
            early_stopping=True,
            random_state=42
        ))
    ])

    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in kfold.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
        scores.append(rmse)

    tune.report(rmse_mean=float(np.mean(scores)))


search_space = {
    "hidden_layer_sizes": tune.choice([(32,), (64,), (64, 32), (128, 64)]),
    "activation": tune.choice(["relu", "tanh"]),
    "alpha": tune.choice([1e-4, 1e-3]),
    "learning_rate_init": tune.choice([1e-3, 5e-3, 1e-2]),
    "batch_size": tune.choice([32, 64]),
}

scheduler = ASHAScheduler(
    metric="rmse_mean",
    mode="min",
    max_t=200,
    grace_period=10,
    reduction_factor=2
)

t0 = perf_counter()

tuner = tune.Tuner(
    train_model,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=scheduler,
        max_concurrent_trials=2   # prevent crash
    )
)

results = tuner.fit()

ray_time = perf_counter() - t0

best_config = results.get_best_result().config

print("Best Ray Params:", best_config)
print(f"Ray Time: {ray_time:.2f}s")

ray.shutdown()


# -------------------------------
# 4. GRID SEARCH
# -------------------------------
print("\n=== GRID SEARCH ===")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(max_iter=200, early_stopping=True, random_state=42))
])

param_grid = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3],
    "mlp__learning_rate_init": [1e-3, 5e-3, 1e-2],
    "mlp__batch_size": [32, 64],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

t1 = perf_counter()
grid.fit(X_train_np, y_train_np)
grid_time = perf_counter() - t1

print("Best Grid Params:", grid.best_params_)
print(f"Grid Time: {grid_time:.2f}s")


# -------------------------------
# 5. FINAL EVALUATION (same test set)
# -------------------------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    rmse_log = np.sqrt(mean_squared_error(y_test, pred))
    mae_log = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    rmse_real = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
    mae_real = mean_absolute_error(np.expm1(y_test), np.expm1(pred))

    return rmse_log, mae_log, r2, rmse_real, mae_real


# Ray model
ray_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(**best_config, max_iter=200, early_stopping=True, random_state=42))
])
ray_model.fit(X_train_np, y_train_np)

ray_metrics = evaluate(ray_model, X_test_np, y_test_np)
grid_metrics = evaluate(grid.best_estimator_, X_test_np, y_test_np)


# -------------------------------
# 6. RESULTS COMPARISON
# -------------------------------
print("\n=== FINAL COMPARISON ===")

print("\n--- Ray ---")
print(f"Time: {ray_time:.2f}s")
print(f"RMSE (£): {ray_metrics[3]:.2f} | MAE (£): {ray_metrics[4]:.2f} | R²: {ray_metrics[2]:.4f}")

print("\n--- GridSearch ---")
print(f"Time: {grid_time:.2f}s")
print(f"RMSE (£): {grid_metrics[3]:.2f} | MAE (£): {grid_metrics[4]:.2f} | R²: {grid_metrics[2]:.4f}")