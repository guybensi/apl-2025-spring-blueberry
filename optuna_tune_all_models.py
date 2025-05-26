# Filename: optuna_tune_all_models.py

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Load and preprocess data
train = pd.read_csv("train.csv")
X = train.drop(columns=["id", "yield"])
y = train["yield"]

# Feature engineering

def create_features(df):
    df = df.copy()
    df["bee_total"] = df["honeybee"] + df["bumbles"] + df["andrena"] + df["osmia"]
    df["bee_to_clonesize"] = df["bee_total"] / df["clonesize"]
    if "MaxOfUpperTRange" in df.columns and "MinOfUpperTRange" in df.columns:
        df["temp_range_upper"] = df["MaxOfUpperTRange"] - df["MinOfUpperTRange"]
    if "MaxOfLowerTRange" in df.columns and "MinOfLowerTRange" in df.columns:
        df["temp_range_lower"] = df["MaxOfLowerTRange"] - df["MinOfLowerTRange"]
    df["fruitmass_per_seed"] = df["fruitmass"] / df["seeds"]
    df["fruit_score"] = df["fruitset"] * df["fruitmass"]
    if "RainingDays" in df.columns and "AverageRainingDays" in df.columns:
        df["raining_ratio"] = df["RainingDays"] / (df["AverageRainingDays"] + 1e-5)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

X = create_features(X)
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Objective function for Optuna
def objective(trial):
    models = {
        "xgb": XGBRegressor(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 300),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
            random_state=42,
            verbosity=0
        ),
        "optuna_hgbr": HistGradientBoostingRegressor(
            learning_rate=0.02048,
            max_iter=661,
            max_depth=9,
            min_samples_leaf=14,
            l2_regularization=4.0155,
            random_state=42
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=trial.suggest_int("gbr_n_estimators", 100, 300),
            max_depth=trial.suggest_int("gbr_max_depth", 3, 8),
            learning_rate=trial.suggest_float("gbr_learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("gbr_subsample", 0.6, 1.0),
            random_state=42
        )
    }

    pred_val = np.zeros(len(X_selected))
    model_weights = {
        "xgb": 0.25,
        "optuna_hgbr": 0.50,
        "gbr": 0.25
    }

    for name, model in models.items():
        fold_val = np.zeros(len(X_selected))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train = y.iloc[train_idx]
            m = clone(model)
            m.fit(X_train, y_train)
            fold_val[val_idx] = m.predict(X_val)
        pred_val += model_weights[name] * fold_val

    return mean_absolute_error(y, pred_val)

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("\nBest MAE:", study.best_value)
print("Best params:", study.best_params)