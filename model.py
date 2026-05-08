#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("processed_aqi_data.csv")
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# -------------------------------
# TIMESTAMP PROCESSING
# -------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["timestamp"])

df["ts_hour"] = df["timestamp"].dt.hour
df["ts_day"] = df["timestamp"].dt.day
df["ts_month"] = df["timestamp"].dt.month
df["ts_year"] = df["timestamp"].dt.year

df = df.sort_values("timestamp").reset_index(drop=True)

# -------------------------------
# ENCODE CATEGORICAL
# -------------------------------
df["location_name"] = df["location_name"].astype("category").cat.codes

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["pollution_index"] = (
    df["pm25"] + df["pm10"] + df["no2"] +
    df["co"] + df["o3"] + df["so2"]
)

# -------------------------------
# CLEAN
# -------------------------------
df = df.dropna(subset=["aqi", "pm25", "pm10", "no2", "co", "o3", "so2"])

print("Dataset size:", df.shape)

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop(columns=[
    "aqi",
    "aqi_category",
    "timestamp",
    "pm25", "pm10", "no2", "co", "o3", "so2"
])

y = df["aqi"]

feature_names = X.columns.tolist()

print("\nFeatures used:")
print(feature_names)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining: 80% | Testing: 20%")

# -------------------------------
# MODELS
# -------------------------------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=60, learning_rate=0.08, verbosity=0, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=60, learning_rate=0.08, verbose=-1, random_state=42)
}

# -------------------------------
# K-FOLD
# -------------------------------
print("\n=== K-Fold Results ===")

tscv = TimeSeriesSplit(n_splits=3)
kfold_scores = {}

for name, model in models.items():
    print(f"Running {name}...")
    scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    kfold_scores[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

# -------------------------------
# TRAIN + TEST PERFORMANCE
# -------------------------------
print("\n=== Test Performance ===")

test_scores = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    test_scores[name] = r2
    predictions[name] = pred

    print(f"\n{name}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

# -------------------------------
# SELECT BEST MODEL
# -------------------------------
best_model_name = max(test_scores, key=test_scores.get)
best_model = models[best_model_name]

print("\n🏆 BEST MODEL SELECTED:", best_model_name)

# -------------------------------
# T-TEST (Best vs RF)
# -------------------------------
print("\n=== T-Test (Best vs Random Forest) ===")

t_stat, p_val = stats.ttest_ind(
    predictions[best_model_name],
    predictions["Random Forest"]
)

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_val:.4f}")

# -------------------------------
# PLOTS (Best Model)
# -------------------------------
best_pred = predictions[best_model_name]

plt.figure()
plt.scatter(y_test, best_pred, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title(f"{best_model_name} - Predicted vs Actual")
plt.savefig("static/pred_vs_actual.png")
plt.close()

residuals = y_test - best_pred

plt.figure()
plt.scatter(best_pred, residuals, alpha=0.5)
plt.axhline(0)
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title(f"{best_model_name} - Residual Plot")
plt.savefig("static/residual_plot.png")
plt.close()

# -------------------------------
# SAVE MODEL
# -------------------------------
best_model.fit(X_train, y_train)

joblib.dump(best_model, "model.pkl")
joblib.dump(feature_names, "features.pkl")
joblib.dump(best_model_name, "model_name.pkl")

print("\nAQI Scale: 0–500")
print("Final Model Used:", best_model_name)

print("\n✅ DONE: Best model selected and saved")

