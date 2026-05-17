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
int cycle_id = 1;
float capacity = 1.85;

const int PIN_VOLTAGE = 34;
const int PIN_CURRENT = 35;
const int PIN_TEMP = 32;

bool fault_overheat = false;
bool fault_sag = false;
bool fault_unstable = false;

void setup() {
  Serial.begin(115200);
  
  pinMode(PIN_VOLTAGE, INPUT);
  pinMode(PIN_CURRENT, INPUT);
  pinMode(PIN_TEMP, INPUT);

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void setup_wifi() {
  Serial.print("\nConnecting to WiFi: ");
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
  String incomingMsg;
  for (unsigned int i = 0; i < length; i++) {
    incomingMsg += (char)message[i];
  }
  
  if (String(topic) == topic_faults) {
    Serial.print("\n>> Dashboard Command Received: ");
    Serial.println(incomingMsg);
    
    if (incomingMsg == "overheat") {
      fault_overheat = true;
      fault_sag = false;
      fault_unstable = false;
      Serial.println("! OVERRIDE: Thermal Runaway Injected via Dashboard");
    } else if (incomingMsg == "voltage_sag") {
      fault_sag = true;
      fault_overheat = false;
      fault_unstable = false;
      Serial.println("! OVERRIDE: Voltage Sag Injected via Dashboard");
    } else if (incomingMsg == "unstable") {
      fault_unstable = true;
      fault_overheat = false;
      fault_sag = false;
      Serial.println("! OVERRIDE: Unstable Current Injected via Dashboard");
    } else if (incomingMsg == "clear") {
      fault_overheat = false;
      fault_sag = false;
      fault_unstable = false;
      Serial.println("* Dashboard Override Cleared. Returned to Manual Wokwi Sliders.");
    }
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "VoltGuard-ESP32-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected!");
      client.subscribe(topic_faults);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds...");
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
    
    float voltage = 2.5 + ((analogRead(PIN_VOLTAGE) / 4095.0) * (4.2 - 2.5));
    float current = -5.0 + ((analogRead(PIN_CURRENT) / 4095.0) * 10.0);
    float temperature = 20.0 + ((analogRead(PIN_TEMP) / 4095.0) * 60.0);

    if (fault_overheat) {
      temperature = 80.0 + ((float)random(-20, 20) / 10.0);
    }
    if (fault_sag) {
      voltage = 2.0 + ((float)random(-5, 5) / 100.0);
    }
    if (fault_unstable) {
      current = ((float)random(-500, 500) / 100.0);
    }

    String payload = "{";
    payload += "\"voltage\":" + String(voltage, 3) + ",";
    payload += "\"current\":" + String(current, 3) + ",";
    payload += "\"temperature\":" + String(temperature, 2) + ",";
    payload += "\"capacity\":" + String(capacity, 3) + ",";
    payload += "\"id_cycle\":" + String(cycle_id);
    payload += "}";

    client.publish(topic_telemetry, payload.c_str());
    
    Serial.print("Telemetry -> ");
    Serial.println(payload);
  }
}
