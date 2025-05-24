import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import optuna

# Load data
train = pd.read_csv("train.csv")
X = train.drop(columns=["id", "yield"])
y = train["yield"]

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

# Define evaluation metric
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Objective function for Optuna
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_iter": trial.suggest_int("max_iter", 300, 1500),  # העלאה – מודל עמוק לומד לאט
        "max_depth": trial.suggest_int("max_depth", 6, 14),    # יותר עומק למורכבות גבוהה
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),  # ננסה עלים גדולים יותר
        "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 10.0, log=True)  # רגולריזציה רחבה
    }


    model = HistGradientBoostingRegressor(**params, random_state=42)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)
    return scores.mean()  # negative MAE

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    print(f"  MAE: {-study.best_value:.3f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model on all data
    final_model = HistGradientBoostingRegressor(**study.best_params, random_state=42)
    final_model.fit(X, y)

    # Optionally save model
    # import joblib
    # joblib.dump(final_model, "best_hist_gbr_model.pkl")
