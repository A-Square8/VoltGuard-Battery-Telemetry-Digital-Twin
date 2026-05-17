import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

df = pd.read_csv('discharge.csv')
df['Health_State'] = np.where(df['Capacity'] > 1.6, 0, np.where(df['Capacity'] > 1.4, 1, 2))

X_xgb = df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'id_cycle']]
y_xgb = df['Health_State']

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

y_pred_xgb = xgb_model.predict(X_xgb)
print("XGBoost Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_xgb, y_pred_xgb))
print("F1 Score (Macro):", f1_score(y_xgb, y_pred_xgb, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_xgb, y_pred_xgb))

iso_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
X_iso = df[iso_features]

with open('iso_scaler.pkl', 'rb') as f:
    iso_scaler = pickle.load(f)

with open('iso_model.pkl', 'rb') as f:
    iso_model = pickle.load(f)

X_iso_scaled = iso_scaler.transform(X_iso)
iso_preds = iso_model.predict(X_iso_scaled)
anomalies = np.sum(iso_preds == -1)
normal = np.sum(iso_preds == 1)

print("\nIsolation Forest Anomaly Detection Evaluation:")
print(f"Total Normal Records (1): {normal}")
print(f"Total Anomalies Detected (-1): {anomalies}")
print(f"Contamination Rate: {anomalies / (anomalies + normal):.4f}")

rul_df = df.groupby('id_cycle').agg({'Capacity': 'mean'}).reset_index()
max_cycle = rul_df['id_cycle'].max()
rul_df['RUL'] = max_cycle - rul_df['id_cycle']

sequence_length = 10
X_lstm = []
y_lstm = []
capacities = rul_df['Capacity'].values
ruls = rul_df['RUL'].values

for i in range(len(capacities) - sequence_length):
    X_lstm.append(capacities[i:i+sequence_length])
    y_lstm.append(ruls[i+sequence_length])

X_lstm = np.array(X_lstm).reshape(-1, sequence_length, 1)
y_lstm = np.array(y_lstm)

lstm_model = load_model('lstm_rul_model.keras')
y_pred_lstm = lstm_model.predict(X_lstm).flatten()

mae = mean_absolute_error(y_lstm, y_pred_lstm)
rmse = np.sqrt(mean_squared_error(y_lstm, y_pred_lstm))

print("\nLSTM Remaining Useful Life (RUL) Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f} cycles")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
