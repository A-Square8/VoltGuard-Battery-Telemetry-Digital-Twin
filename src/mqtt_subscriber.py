import json
import time
import paho.mqtt.client as mqtt
from feature_extractor import FeatureExtractor

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TELEMETRY_TOPIC = "voltguard/telemetry"
FAULTS_TOPIC = "voltguard/faults/inject"

extractor = FeatureExtractor(window_size=5)

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT Broker")
        client.subscribe(TELEMETRY_TOPIC)
    else:
        print(f"Failed to connect, return code {reason_code}")

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
        voltage = data.get("voltage", 0.0)
        current = data.get("current", 0.0)
        temp = data.get("temperature", 0.0)
        capacity = data.get("capacity", 0.0)
        cycle = data.get("id_cycle", 0)
        
        timestamp = time.time()
        extractor.update(voltage, current, temp, timestamp)
        features = extractor.extract_features()
        
        print(f"[RAW] V: {voltage:.2f} | I: {current:.2f} | T: {temp:.2f}")
        if features:
            print(f"[FEATURES] dV/dt: {features['dv_dt']:.4f} | dT/dt: {features['dt_dt']:.4f} | Spike: {features['temp_spike']} | Instability: {features['current_instability']:.4f}")
            
    except json.JSONDecodeError:
        print(f"[RAW ERROR] {payload}")

def start_subscriber():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to {MQTT_BROKER}")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("Disconnecting")
        client.disconnect()

if __name__ == "__main__":
    start_subscriber()
