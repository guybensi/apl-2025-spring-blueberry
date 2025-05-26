# Filename: optimize_blend_weights.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop(columns=["id", "yield"])
y = train["yield"]
X_test = test.drop(columns=["id"])

# Feature Engineering
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
X_test = create_features(X_test)
X_test = X_test[X.columns]

# Feature selection
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

# Define base models
base_models = {
    "xgb": XGBRegressor(
        n_estimators=116,
        max_depth=3,
        learning_rate=0.06147,
        subsample=0.9483,
        colsample_bytree=0.8291,
        reg_lambda=4.2998,
        reg_alpha=4.6666,
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
        n_estimators=127,
        max_depth=8,
        learning_rate=0.04767,
        subsample=0.6528,
        random_state=42
    ),
}


# KFold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions
model_preds_val = {}
model_preds_test = {}

for name, model in base_models.items():
    print(f"Fitting model: {name}")
    fold_val = np.zeros(len(X_selected))
    fold_test = np.zeros((len(X_test_selected), 5))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train = y.iloc[train_idx]
        m = clone(model)
        m.fit(X_train, y_train)
        fold_val[val_idx] = m.predict(X_val)
        fold_test[:, fold] = m.predict(X_test_selected)

    model_preds_val[name] = fold_val
    model_preds_test[name] = fold_test.mean(axis=1)

# Grid search over ensemble weights
best_mae = float("inf")
best_weights = None

for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0: continue

        pred_val = (
            w1 * model_preds_val["xgb"] +
            w2 * model_preds_val["optuna_hgbr"] +
            w3 * model_preds_val["gbr"]
        )
        mae = mean_absolute_error(y, pred_val)

        if mae < best_mae:
            best_mae = mae
            best_weights = (w1, w2, w3)

print(f"\nBest Weights:")
print(f"xgb = {best_weights[0]:.3f}")
print(f"optuna_hgbr = {best_weights[1]:.3f}")
print(f"gbr = {best_weights[2]:.3f}")
print(f"Best MAE: {best_mae:.3f}")
