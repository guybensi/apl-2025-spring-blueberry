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
    
    # חישובים שמבוססים רק על עמודות שעדיין קיימות
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

# Feature selection (use all remaining features)
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

# Best ensemble weights (same as before)
model_weights = {
    "xgb": 0.2515,
    "optuna_hgbr": 0.5263,
    "gbr": 0.2222
}

# Base models (reduce n_estimators to 100 for speed)
base_models = {
    "xgb": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0),
    "optuna_hgbr": HistGradientBoostingRegressor(
        learning_rate=0.0067,
        max_iter=700,
        max_depth=7,
        min_samples_leaf=20,
        l2_regularization=0.583,
        random_state=42
    ),
    "gbr": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
}

# Cross-validation prediction
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred_val = np.zeros(len(X_selected))
pred_test = np.zeros(len(X_test_selected))

for name, model in base_models.items():
    print(f"Training base model: {name}")
    fold_val = np.zeros(len(X_selected))
    fold_test = np.zeros((len(X_test_selected), 5))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train = y.iloc[train_idx]
        m = clone(model)
        m.fit(X_train, y_train)
        fold_val[val_idx] = m.predict(X_val)
        fold_test[:, fold] = m.predict(X_test_selected)

    pred_val += model_weights[name] * fold_val
    pred_test += model_weights[name] * fold_test.mean(axis=1)

# Final evaluation
mae = mean_absolute_error(y, pred_val)
print(f"\nFinal Blended MAE: {mae:.3f}")

# Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "yield": pred_test
})
submission.to_csv("submission_blend_final.csv", index=False)
