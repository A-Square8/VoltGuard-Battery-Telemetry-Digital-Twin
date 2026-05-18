

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("VOLTGUARD ML PIPELINE — TRAINING")
print("=" * 60)


df = pd.read_csv('discharge.csv')
print(f"\nDataset loaded: {len(df):,} records, {len(df.columns)} features")
print(f"Cycles: {df['id_cycle'].nunique()} | Batteries: {df['Battery'].nunique()}")
print(f"Voltage range: {df['Voltage_measured'].min():.2f} - {df['Voltage_measured'].max():.2f} V")
print(f"Temperature range: {df['Temperature_measured'].min():.1f} - {df['Temperature_measured'].max():.1f} °C")

# Model 1: XGBoost Health Classifier
print("\n" + "=" * 60)
print("MODEL 1: XGBoost Health State Classifier")
print("=" * 60)

df['Health_State'] = np.where(df['Capacity'] > 1.6, 0,
                     np.where(df['Capacity'] > 1.4, 1, 2))

class_names = ['HEALTHY (>1.6Ah)', 'DEGRADED (1.4-1.6Ah)', 'CRITICAL (<1.4Ah)']
print(f"\nClass distribution:")
for i, name in enumerate(class_names):
    count = (df['Health_State'] == i).sum()
    print(f"  {name}: {count:,} ({count/len(df)*100:.1f}%)")

X_xgb = df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'id_cycle']]
y_xgb = df['Health_State']

X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb)

xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

print("Feature Importance:")
for feat, imp in sorted(zip(X_xgb.columns, xgb_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("Saved: xgb_model.pkl")

# Model 2: Isolation Forest Anomaly Detection
print("\n" + "=" * 60)
print("MODEL 2: Isolation Forest Anomaly Detector")
print("=" * 60)

iso_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
X_iso = df[iso_features]
iso_scaler = MinMaxScaler()
X_iso_scaled = iso_scaler.fit_transform(X_iso)

print(f"\nScaler ranges (used for anomaly threshold):")
for feat, mn, mx in zip(iso_features, iso_scaler.data_min_, iso_scaler.data_max_):
    print(f"  {feat}: [{mn:.3f}, {mx:.3f}]")

iso_model = IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
iso_model.fit(X_iso_scaled)

iso_preds = iso_model.predict(X_iso_scaled)
anomalies = (iso_preds == -1).sum()
print(f"\nTraining anomalies detected: {anomalies:,} / {len(df):,} ({anomalies/len(df)*100:.2f}%)")
print(f"Any value outside these ranges will be flagged as anomaly.")

with open('iso_model.pkl', 'wb') as f:
    pickle.dump(iso_model, f)
with open('iso_scaler.pkl', 'wb') as f:
    pickle.dump(iso_scaler, f)
print("Saved: iso_model.pkl, iso_scaler.pkl")

# Model 3: LSTM RUL Forecaster
print("\n" + "=" * 60)
print("MODEL 3: LSTM Remaining Useful Life Forecaster")
print("=" * 60)

rul_df = df.groupby('id_cycle').agg({'Capacity': 'mean'}).reset_index()
max_cycle = rul_df['id_cycle'].max()
rul_df['RUL'] = max_cycle - rul_df['id_cycle']
print(f"\nCapacity degradation: {rul_df['Capacity'].iloc[0]:.3f} Ah -> {rul_df['Capacity'].iloc[-1]:.3f} Ah")
print(f"Total cycles: {max_cycle}")

seq_len = 10
X_lstm, y_lstm = [], []
caps = rul_df['Capacity'].values
ruls = rul_df['RUL'].values
for i in range(len(caps) - seq_len):
    X_lstm.append(caps[i:i+seq_len])
    y_lstm.append(ruls[i+seq_len])
X_lstm = np.array(X_lstm).reshape(-1, seq_len, 1)
y_lstm = np.array(y_lstm)
print(f"Sequences created: {len(X_lstm)}")

split = int(len(X_lstm) * 0.8)
X_tr, X_te = X_lstm[:split], X_lstm[split:]
y_tr, y_te = y_lstm[:split], y_lstm[split:]

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_len, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = lstm_model.fit(X_tr, y_tr, epochs=50, batch_size=16, validation_data=(X_te, y_te),
                         callbacks=[early_stop], verbose=1)

y_pred_lstm = lstm_model.predict(X_te).flatten()
mae = np.mean(np.abs(y_te - y_pred_lstm))
rmse = np.sqrt(np.mean((y_te - y_pred_lstm)**2))
print(f"\nTest MAE: {mae:.2f} cycles")
print(f"Test RMSE: {rmse:.2f} cycles")

lstm_model.save('lstm_rul_model.keras')
print("Saved: lstm_rul_model.keras")
print("\n" + "=" * 60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 60)
