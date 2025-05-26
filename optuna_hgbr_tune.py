import pandas as pd
import numpy as np
import optuna
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

# Load and preprocess
train = pd.read_csv("train.csv")
X = train.drop(columns=["id", "yield"])
y = train["yield"]

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

# Remove mismatch if needed (like in main file)
# feature intersection between train/test is not needed here since it's only train

# Feature selection
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)

# Optuna objective
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.04, log=True),
        "max_iter": trial.suggest_int("max_iter", 550, 700),
        "max_depth": trial.suggest_int("max_depth", 8, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 8, 15),
        "l2_regularization": trial.suggest_float("l2_regularization", 3.5, 4.5),
        "random_state": 42
    }


    model = HistGradientBoostingRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for train_idx, val_idx in kf.split(X_selected):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)

    return np.mean(maes)

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print best parameters
print("Best MAE:", study.best_value)
print("Best params:", study.best_params)
