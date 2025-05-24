import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import optuna

# Load data
train = pd.read_csv("train.csv")
X = train.drop(columns=["id", "yield", "Row#"])
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

# Evaluation metric
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Optuna objective function
def objective(trial):
    k = trial.suggest_int("k_best", 10, X.shape[1])

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_iter": trial.suggest_int("max_iter", 300, 1500),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 10.0, log=True)
    }

    selector = SelectKBest(score_func=f_regression, k=k)
    model = HistGradientBoostingRegressor(**params, random_state=42)
    pipeline = Pipeline([
        ("select", selector),
        ("model", model)
    ])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring=mae_scorer)
    return scores.mean()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    print(f"  MAE: {-study.best_value:.3f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Final training
    selector = SelectKBest(score_func=f_regression, k=study.best_params["k_best"])
    model = HistGradientBoostingRegressor(
        learning_rate=study.best_params["learning_rate"],
        max_iter=study.best_params["max_iter"],
        max_depth=study.best_params["max_depth"],
        min_samples_leaf=study.best_params["min_samples_leaf"],
        l2_regularization=study.best_params["l2_regularization"],
        random_state=42
    )
    pipeline = Pipeline([
        ("select", selector),
        ("model", model)
    ])
    pipeline.fit(X, y)

    # Optionally save model
    # import joblib
    # joblib.dump(pipeline, "best_model_fs.pkl")
