
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

print("=" * 60)
print("VOLTGUARD ML PIPELINE — EVALUATION")
print("=" * 60)

df = pd.read_csv('discharge.csv')
df['Health_State'] = np.where(df['Capacity'] > 1.6, 0, np.where(df['Capacity'] > 1.4, 1, 2))

# XGBoost
print("\n" + "-" * 40)
print("MODEL 1: XGBoost Health Classifier")
print("-" * 40)

with open('xgb_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
X_xgb = df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'id_cycle']]
y_xgb = df['Health_State']
y_pred = xgb_model.predict(X_xgb)

print(f"Accuracy: {accuracy_score(y_xgb, y_pred):.4f}")
print(f"F1 (Macro): {f1_score(y_xgb, y_pred, average='macro'):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_xgb, y_pred)}")
print(f"\n{classification_report(y_xgb, y_pred, target_names=['HEALTHY','DEGRADED','CRITICAL'])}")

# Isolation Forest
print("-" * 40)
print("MODEL 2: Isolation Forest Anomaly Detector")
print("-" * 40)

with open('iso_model.pkl', 'rb') as f: iso_model = pickle.load(f)
with open('iso_scaler.pkl', 'rb') as f: iso_scaler = pickle.load(f)

X_iso = df[['Voltage_measured', 'Current_measured', 'Temperature_measured']]
X_iso_scaled = iso_scaler.transform(X_iso)
iso_preds = iso_model.predict(X_iso_scaled)
anom = (iso_preds == -1).sum()
norm = (iso_preds == 1).sum()

print(f"Normal: {norm:,} | Anomalies: {anom:,}")
print(f"Contamination rate: {anom/(anom+norm):.4f}")

# Verify fault detection scenarios
print("\nFault Injection Verification:")
scenarios = [
    ("Normal (V=3.7, I=-2.0, T=28)", [3.7, -2.0, 28.0]),
    ("Thermal Runaway (V=3.5, I=-2.5, T=80)", [3.5, -2.5, 80.0]),
    ("Voltage Sag (V=1.0, I=-3.0, T=30)", [1.0, -3.0, 30.0]),
    ("Unstable Current (V=3.6, I=5.0, T=40)", [3.6, 5.0, 40.0]),
]
for name, vals in scenarios:
    x = iso_scaler.transform([vals])
    pred = iso_model.predict(x)[0]
    label = "ANOMALY" if pred == -1 else "NORMAL"
    print(f"  {name} -> {label}")

# LSTM RUL
print("\n" + "-" * 40)
print("MODEL 3: LSTM RUL Forecaster")
print("-" * 40)

lstm_model = load_model('lstm_rul_model.keras')
rul_df = df.groupby('id_cycle').agg({'Capacity': 'mean'}).reset_index()
max_cycle = rul_df['id_cycle'].max()
rul_df['RUL'] = max_cycle - rul_df['id_cycle']

seq_len = 10
X_lstm, y_lstm = [], []
caps = rul_df['Capacity'].values
ruls = rul_df['RUL'].values
for i in range(len(caps) - seq_len):
    X_lstm.append(caps[i:i+seq_len])
    y_lstm.append(ruls[i+seq_len])
X_lstm = np.array(X_lstm).reshape(-1, seq_len, 1)
y_lstm = np.array(y_lstm)

y_pred_lstm = lstm_model.predict(X_lstm).flatten()
mae = mean_absolute_error(y_lstm, y_pred_lstm)
rmse = np.sqrt(mean_squared_error(y_lstm, y_pred_lstm))
print(f"MAE: {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
