#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "Wokwi-GUEST";
const char* password = "";

const char* mqtt_server = "broker.hivemq.com";
const char* topic_telemetry = "voltguard/telemetry";
const char* topic_faults = "voltguard/faults/inject";

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
int cycle_id = 80;
float capacity = 1.56;
int step_count = 0;

// Fault injection flags
bool fault_overheat = false;
bool fault_sag = false;
bool fault_unstable = false;

void setup() {
  Serial.begin(115200);
  Serial.println("\n========================================");
  Serial.println("  VoltGuard ESP32 — Battery Digital Twin");
  Serial.println("========================================");
  Serial.println("Mode: NORMAL OPERATION");
  Serial.println("Generating NASA-range telemetry...\n");

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void setup_wifi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
}

void callback(char* topic, byte* message, unsigned int length) {
  String cmd;
  for (unsigned int i = 0; i < length; i++) {
    cmd += (char)message[i];
  }
  
  if (String(topic) == topic_faults) {
    Serial.println("\n----------------------------------------");
    Serial.print(">> DASHBOARD COMMAND: ");
    Serial.println(cmd);
    
    if (cmd == "overheat") {
      fault_overheat = true;
      fault_sag = false;
      fault_unstable = false;
      Serial.println("!! FAULT INJECTED: Thermal Runaway");
      Serial.println("   Temperature elevating to 75-85C");
      Serial.println("   Voltage dropping under thermal stress");
      Serial.println("   Expected: Isolation Forest -> ANOMALY");
    } else if (cmd == "voltage_sag") {
      fault_sag = true;
      fault_overheat = false;
      fault_unstable = false;
      Serial.println("!! FAULT INJECTED: Voltage Sag");
      Serial.println("   Voltage collapsing to 0.5-1.5V");
      Serial.println("   Expected: XGBoost -> CRITICAL");
    } else if (cmd == "unstable") {
      fault_unstable = true;
      fault_overheat = false;
      fault_sag = false;
      Serial.println("!! FAULT INJECTED: Current Instability");
      Serial.println("   Current oscillating -8A to +8A");
      Serial.println("   Expected: Isolation Forest -> ANOMALY");
    } else if (cmd == "clear" || cmd == "normal") {
      fault_overheat = false;
      fault_sag = false;
      fault_unstable = false;
      Serial.println("** FAULTS CLEARED: Normal operation restored");
      Serial.println("   Readings returning to NASA dataset range");
    }
    Serial.println("----------------------------------------\n");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("MQTT connecting...");
    String clientId = "VoltGuard-ESP32-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected!");
      client.subscribe(topic_faults);
      Serial.println("Subscribed to fault injection topic.");
    } else {
      Serial.print("failed (rc=");
      Serial.print(client.state());
      Serial.println(") retrying in 5s...");
      delay(5000);
    }
  }
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg > 1000) {
    lastMsg = now;
    step_count++;

    // ---- DEFAULT: Generate values within NASA dataset range ----
    // NASA ranges: V=1.74-4.04  I=-2.03 to -1.98  T=22.4-42.1
    float t_phase = step_count * 0.05;
    float voltage = 3.7 + 0.25 * sin(t_phase * 0.3) + ((float)random(-10, 10) / 1000.0);
    float current = -2.0 + 0.015 * sin(t_phase * 0.7) + ((float)random(-5, 5) / 1000.0);
    float temperature = 28.0 + 4.0 * sin(t_phase * 0.15) + ((float)random(-10, 10) / 100.0);

    // Slow capacity fade (realistic aging)
    if (step_count % 60 == 0) {
      cycle_id++;
      capacity = max(1.15f, capacity - 0.003f);
    }

    // ---- FAULT OVERRIDES ----
    String mode_tag = "NORMAL";
    
    if (fault_overheat) {
      mode_tag = "OVERHEAT";
      temperature = 75.0 + ((float)random(0, 100) / 10.0);   // 75-85C
      voltage = max(2.0f, voltage - 1.2f);                     // Voltage drops
      current = current - 0.5 + ((float)random(-30, 30) / 100.0);
    }
    
    if (fault_sag) {
      mode_tag = "SAG";
      voltage = 0.5 + ((float)random(0, 100) / 100.0);        // 0.5-1.5V
      capacity = max(0.5f, capacity * 0.4f);
    }
    
    if (fault_unstable) {
      mode_tag = "UNSTABLE";
      current = ((float)random(-800, 800) / 100.0);            // -8 to +8A
      temperature = temperature + abs(current) * 1.5;
    }

    String payload = "{";
    payload += "\"voltage\":" + String(voltage, 3) + ",";
    payload += "\"current\":" + String(current, 3) + ",";
    payload += "\"temperature\":" + String(temperature, 2) + ",";
    payload += "\"capacity\":" + String(capacity, 3) + ",";
    payload += "\"id_cycle\":" + String(cycle_id);
    payload += "}";

    client.publish(topic_telemetry, payload.c_str());
    
    Serial.print("[");
    Serial.print(mode_tag);
    Serial.print("] V=");
    Serial.print(voltage, 2);
    Serial.print(" I=");
    Serial.print(current, 2);
    Serial.print(" T=");
    Serial.print(temperature, 1);
    Serial.print(" Cyc=");
    Serial.println(cycle_id);
  }
}
