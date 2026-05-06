#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("delhi_aqi_data.csv")
df.columns = df.columns.str.strip()

# Convert to numeric
cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'Ozone', 'Month', 'Days', 'Holidays_Count', 'AQI']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# -------------------------------
# FEATURES (MATCHES APP INPUTS)
# -------------------------------
X = df[['PM2.5', 'PM10', 'NO2', 'CO', 'Ozone', 'Month', 'Days', 'Holidays_Count']]
y = df['AQI']

print("Dataset size:", X.shape)

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{name}")
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")

print("\n=== MODEL REPORT ===")
evaluate("Random Forest", y_test, rf_pred)
evaluate("Linear Regression", y_test, lr_pred)

# -------------------------------
# SHAP
# -------------------------------
print("\nGenerating SHAP plot...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("static/shap_summary.png")

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(rf, "model.pkl")

print("\n✅ Model fixed and saved!")

