import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

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

# Define MLP model
input_dim = X.shape[1]
def create_mlp():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mae')
    return model

# Base models (add MLP as one of them)
base_models = [
    ("xgb", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)),
    ("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ("optuna_hgbr", HistGradientBoostingRegressor(
        learning_rate=0.0067,
        max_iter=700,
        max_depth=7,
        min_samples_leaf=20,
        l2_regularization=0.583,
        random_state=42
    )),
    ("mlp_base", KerasRegressor(build_fn=create_mlp, epochs=30, batch_size=32, verbose=0))
]

# Prepare for stacking
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
        if name == "mlp_base":
            m = KerasRegressor(build_fn=create_mlp, epochs=30, batch_size=32, verbose=0)
            m.fit(X_train, y_train, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        else:
            m.fit(X_train, y_train)
        fold_preds[val_idx] = m.predict(X_val)
        fold_test_preds[:, fold] = m.predict(X_test)

    oof_predictions[:, i] = fold_preds
    test_predictions[:, i] = fold_test_preds.mean(axis=1)

# MLP meta-model
print("\nTraining MLP meta-model")
meta_model = create_mlp()
meta_model.fit(oof_predictions, y, epochs=50, batch_size=16, verbose=0,
               callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
final_preds = meta_model.predict(test_predictions).flatten()

# Evaluate
meta_oof_preds = meta_model.predict(oof_predictions).flatten()
mae = mean_absolute_error(y, meta_oof_preds)
print(f"\nBlended MAE with MLP meta-model: {mae:.3f}")

# Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "yield": final_preds
})
submission.to_csv("submission_blend_mlp.csv", index=False)
