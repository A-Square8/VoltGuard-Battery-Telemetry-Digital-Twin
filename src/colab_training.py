import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv('discharge.csv')

df['Health_State'] = np.where(df['Capacity'] > 1.6, 0, 
                     np.where(df['Capacity'] > 1.4, 1, 2))

X_xgb = df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'id_cycle']]
y_xgb = df['Health_State']

xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_xgb, y_xgb)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

iso_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
X_iso = df[iso_features]
iso_scaler = MinMaxScaler()
X_iso_scaled = iso_scaler.fit_transform(X_iso)

iso_model = IsolationForest(contamination=0.01, random_state=42)
iso_model.fit(X_iso_scaled)

with open('iso_model.pkl', 'wb') as f:
    pickle.dump(iso_model, f)

with open('iso_scaler.pkl', 'wb') as f:
    pickle.dump(iso_scaler, f)

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

lstm_model = Sequential()
lstm_model.add(LSTM(32, activation='relu', input_shape=(sequence_length, 1)))
lstm_model.add(Dense(16, activation='relu'))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=1)

lstm_model.save('lstm_rul_model.keras')
