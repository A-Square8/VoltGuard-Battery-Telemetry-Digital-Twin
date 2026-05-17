import json
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TELEMETRY_TOPIC = "voltguard/telemetry"
FAULTS_TOPIC = "voltguard/faults/inject"

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
        print(f"[TELEMETRY] V: {voltage:.2f}V | I: {current:.2f}A | T: {temp:.2f}C | Cap: {capacity:.2f}Ah | Cycle: {cycle}")
    except json.JSONDecodeError:
        print(f"[RAW] {payload}")

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
