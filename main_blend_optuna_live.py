import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
import optuna

# Load and prepare data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop(columns=["id", "yield", "Row#"])
y = train["yield"]
X_test = test.drop(columns=["id", "Row#"])

# Feature Engineering
def create_features(df):
    df = df.copy()
    df["bee_total"] = df["honeybee"] + df["bumbles"] + df["andrena"] + df["osmia"]
    df["bee_to_clonesize"] = df["bee_total"] / df["clonesize"]
    df["temp_range_upper"] = df["MaxOfUpperTRange"] - df["MinOfUpperTRange"]
    df["temp_range_lower"] = df["MaxOfLowerTRange"] - df["MinOfLowerTRange"]
    df["fruitmass_per_seed"] = df["fruitmass"] / df["seeds"]
    df["fruit_score"] = df["fruitset"] * df["fruitmass"]
    df["raining_ratio"] = df["RainingDays"] / (df["AverageRainingDays"] + 1e-5)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

X = create_features(X)
X_test = create_features(X_test)

# Train Optuna-tuned HGBR dynamically
def optuna_model(X, y):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_iter": trial.suggest_int("max_iter", 300, 1000),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 10.0, log=True),
        }
        model = HistGradientBoostingRegressor(**params, random_state=42)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            score = mean_absolute_error(y.iloc[val_idx], preds)
            scores.append(score)
        return -np.mean(scores)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10)
    print("\nBest Optuna Model:", study.best_params)
    return HistGradientBoostingRegressor(**study.best_params, random_state=42)

optuna_model_instance = optuna_model(X, y)

# Base models (including dynamic Optuna model)
base_models = [
    ("xgb", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)),
    ("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ("optuna_dynamic", optuna_model_instance)
]

# Meta model
meta_model = Ridge(alpha=1.0)

# Out-of-fold predictions for training meta model
oof_predictions = np.zeros((X.shape[0], len(base_models)))
test_predictions = np.zeros((X_test.shape[0], len(base_models)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (name, model) in enumerate(base_models):
    print(f"Training base model: {name}")
    fold_preds = np.zeros(X.shape[0])
    fold_test_preds = np.zeros((X_test.shape[0], 5))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        fold_preds[val_idx] = m.predict(X_val)
        fold_test_preds[:, fold] = m.predict(X_test)

    oof_predictions[:, i] = fold_preds
    test_predictions[:, i] = fold_test_preds.mean(axis=1)

# Train meta model
meta_model.fit(oof_predictions, y)
final_preds = meta_model.predict(test_predictions)

# Evaluate on out-of-fold predictions
meta_oof_preds = meta_model.predict(oof_predictions)
mae = mean_absolute_error(y, meta_oof_preds)
print(f"\nBlended MAE on validation: {mae:.3f}")

# Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "yield": final_preds
})
submission.to_csv("submission_blend_optuna_live.csv", index=False)
