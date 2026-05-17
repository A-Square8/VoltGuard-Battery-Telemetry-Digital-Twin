import gradio as gr
import numpy as np
import pandas as pd
import pickle
import time
import os
import sys
import threading
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
sys.path.insert(0, SRC_DIR)
from feature_extractor import FeatureExtractor
from data_simulator import DatasetSimulator

# --- Models ---
with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'rb') as f: xgb_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'iso_model.pkl'), 'rb') as f: iso_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'iso_scaler.pkl'), 'rb') as f: iso_scaler = pickle.load(f)
try:
    from tensorflow.keras.models import load_model
    lstm_model = load_model(os.path.join(MODELS_DIR, 'lstm_rul_model.keras'))
except Exception: lstm_model = None

# --- State ---
sim = DatasetSimulator()
extractor = FeatureExtractor(window_size=5)
raw_hist = deque(maxlen=10)
feat_hist = deque(maxlen=10)
preds = {"health":"---","anomaly":"---","rul":0,"score":0}
source = {"mode": None}

# --- MQTT ---
mqtt_ok = False; mqtt_client = None
BROKER="broker.hivemq.com"; FAULT_T="voltguard/faults/inject"; TELE_T="voltguard/telemetry"
def _mqtt_msg(client, ud, msg):
    import json
    try:
        p=json.loads(msg.payload.decode())
        _process(p["voltage"],p["current"],p["temperature"],p.get("capacity",1.85),p.get("id_cycle",1))
    except: pass
def _mqtt_start():
    global mqtt_ok, mqtt_client
    try:
        import paho.mqtt.client as mqtt
        mqtt_client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        def oc(c,u,f,rc,pr):
            global mqtt_ok
            if rc==0: mqtt_ok=True; c.subscribe(TELE_T)
        mqtt_client.on_connect=oc; mqtt_client.on_message=_mqtt_msg
        mqtt_client.connect(BROKER,1883,60); mqtt_client.loop_forever()
    except: pass
threading.Thread(target=_mqtt_start, daemon=True).start()

def _process(v, c, t, cap, cyc):
    ts = datetime.now().strftime("%H:%M:%S")
    raw_hist.append({"Time":ts,"Voltage":f"{v:.3f}","Current":f"{c:.3f}","Temp":f"{t:.2f}","Cap":f"{cap:.3f}","Cycle":cyc})
    extractor.update(v, c, t, cap, time.time())
    feats = extractor.extract_features()
    if feats:
        feat_hist.append({"Time":ts,"dV/dt":f"{feats['dv_dt']:.5f}","V_var":f"{feats['v_var']:.5f}",
            "I_instab":f"{feats['current_instability']:.5f}","FFT":f"{feats['fft_v_mean']:.5f}","T_spike":feats['temp_spike']})
        x=pd.DataFrame([[v,c,t,cyc]],columns=['Voltage_measured','Current_measured','Temperature_measured','id_cycle'])
        hp=xgb_model.predict(x)[0]; preds["health"]=["HEALTHY","DEGRADED","CRITICAL"][hp]; preds["score"]=[99,65,20][hp]
        xi=pd.DataFrame([[v,c,t]],columns=['Voltage_measured','Current_measured','Temperature_measured'])
        ip=iso_model.predict(iso_scaler.transform(xi))[0]; preds["anomaly"]="THERMAL RUNAWAY" if ip==-1 else "NORMAL"
    if lstm_model and len(extractor.capacity_buffer)==10:
        xl=np.array(extractor.capacity_buffer).reshape(1,10,1); preds["rul"]=max(0,int(lstm_model.predict(xl,verbose=0)[0][0]))

# --- Handlers ---
def _clear_state():
    """Reset all history when switching data sources."""
    raw_hist.clear()
    feat_hist.clear()
    preds.update({"health":"---","anomaly":"---","rul":0,"score":0})
    extractor.__init__(window_size=5)
    sim.event_log.clear()
    sim.mode = "normal"

def start_sim():
    _clear_state()
    source["mode"]="simulator"; sim.start(); return build()
def start_wokwi():
    _clear_state()
    source["mode"]="wokwi"
    sim._log("SYSTEM","Waiting for Wokwi ESP32 data via MQTT...")
    sim._log("SYSTEM","Wokwi project opened in new tab. Start the simulation there.")
    sim._log("SYSTEM",f"MQTT Broker: {BROKER} | Topic: {TELE_T}")
    return build()
def do_fault(mode):
    if source["mode"]=="simulator": sim.set_mode(mode)
    if mqtt_client and mqtt_ok:
        try: mqtt_client.publish(FAULT_T, mode if mode!="normal" else "clear")
        except: pass
    return build()
def tick():
    if source["mode"]=="simulator" and sim.running:
        r=sim.generate_reading()
        if r: _process(r["voltage"],r["current"],r["temperature"],r["capacity"],r["id_cycle"])
    return build()

# --- HTML Helpers ---
def _tbl(rows, cols, title):
    if not rows:
        return f'<div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:12px;padding:1.5rem;min-height:260px;display:flex;flex-direction:column;justify-content:center;align-items:center;"><span style="color:#D4AF37;font-size:0.65rem;font-weight:700;letter-spacing:2px;">{title}</span><p style="color:#444;font-size:0.8rem;margin-top:0.5rem;">Waiting for data...</p></div>'
    h=''.join(f'<th style="padding:0.35rem 0.5rem;color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:1.5px;border-bottom:1px solid #222;text-align:left;">{c}</th>' for c in cols)
    b=''.join('<tr>'+''.join(f'<td style="padding:0.3rem 0.5rem;color:#ccc;font-family:JetBrains Mono,monospace;font-size:0.75rem;border-bottom:1px solid #111;">{r.get(c,"")}</td>' for c in cols)+'</tr>' for r in reversed(list(rows)))
    return f'<div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:12px;overflow:hidden;min-height:260px;"><div style="padding:0.8rem 1rem 0.4rem;"><span style="color:#D4AF37;font-size:0.65rem;font-weight:700;letter-spacing:2px;">{title}</span></div><div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div></div>'

def _w(label, val, unit, icon):
    return f'<div style="background:#121212;border:1px solid #222;border-top:3px solid #D4AF37;border-radius:12px;padding:1rem;"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;"><span style="color:#888;font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">{label}</span><span class="material-symbols-outlined" style="color:#D4AF37;font-size:1.2rem;">{icon}</span></div><div style="font-family:JetBrains Mono,monospace;font-size:2.5rem;font-weight:700;color:#fff;line-height:1;">{val}<span style="font-size:0.9rem;color:#666;margin-left:4px;">{unit}</span></div></div>'

def _fault_panel():
    desc = sim.get_fault_panel()
    is_fault = sim.mode != "normal"
    border = "#FF3366" if is_fault else "#222"
    icon_color = "#FF3366" if is_fault else "#00FFCC"
    icon = "warning" if is_fault else "check_circle"
    bg = "#1a0000" if is_fault else "#001a0d"
    return f'''<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:1.2rem;margin-bottom:1.5rem;">
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">
      <span class="material-symbols-outlined" style="color:{icon_color};font-size:1.3rem;">{icon}</span>
      <span style="color:{icon_color};font-size:0.8rem;font-weight:700;letter-spacing:1.5px;">{desc["title"]}</span>
    </div>
    <p style="color:#999;font-size:0.82rem;line-height:1.6;margin:0 0 0.5rem;">{desc["detail"]}</p>
    <p style="color:#666;font-family:JetBrains Mono,monospace;font-size:0.7rem;margin:0;">{desc["expected"]}</p>
    </div>'''

def build():
    if source["mode"] is None:
        return '<div style="background:#050505;padding:3rem;text-align:center;"><p style="color:#444;font-size:1rem;font-weight:600;letter-spacing:1px;">Select a data source above to begin monitoring.</p></div>'
    last=list(raw_hist)[-1] if raw_hist else None
    v=float(last["Voltage"]) if last else 0; c=float(last["Current"]) if last else 0; t=float(last["Temp"]) if last else 0
    h,a,rul,hs=preds["health"],preds["anomaly"],preds["rul"],preds["score"]
    hc="#00FFCC" if h=="HEALTHY" else ("#FFCC00" if h=="DEGRADED" else "#FF3366")
    ac="#FF3366" if a=="THERMAL RUNAWAY" else "#00FFCC"
    md=sim.mode if source["mode"]=="simulator" else "wokwi"
    ml=DatasetSimulator.FAULT_DESCRIPTIONS.get(md,{}).get("title","WOKWI MQTT STREAM") if source["mode"]=="simulator" else "WOKWI MQTT STREAM"
    dot="#00FFCC" if md=="normal" or md=="wokwi" else "#FF3366"
    sl="SIMULATOR (NASA DATASET)" if source["mode"]=="simulator" else "WOKWI (MQTT)"
    logs=sim.get_log()
    lh=''.join(f'<div style="padding:0.15rem 0;color:#888;font-family:JetBrains Mono,monospace;font-size:0.7rem;border-bottom:1px solid #0a0a0a;">{l}</div>' for l in reversed(logs[-8:]))
    rc=["Time","Voltage","Current","Temp","Cap","Cycle"]; fc=["Time","dV/dt","V_var","I_instab","FFT","T_spike"]
    return f'''<div style="background:#050505;color:#fff;font-family:Inter,sans-serif;padding:1.5rem 2rem;"><div style="max-width:1400px;margin:0 auto;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;padding:0.7rem 1rem;background:#0a0a0a;border:1px solid #1a1a1a;border-radius:10px;">
    <div style="display:flex;align-items:center;gap:0.6rem;"><div style="width:8px;height:8px;border-radius:50%;background:{dot};box-shadow:0 0 6px {dot};"></div><span style="color:#fff;font-weight:700;font-size:0.75rem;letter-spacing:1px;">{ml}</span></div>
    <span style="color:#D4AF37;font-size:0.65rem;font-weight:700;letter-spacing:1.5px;">{sl}</span>
  </div>
  <div style="background:#080808;border:1px solid #1a1a1a;border-radius:10px;padding:0.7rem 0.8rem;margin-bottom:1.5rem;max-height:140px;overflow-y:auto;">
    <div style="color:#D4AF37;font-size:0.55rem;font-weight:700;letter-spacing:2px;margin-bottom:0.3rem;">EVENT LOG</div>{lh}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;">
    {_tbl(raw_hist, rc, "LIVE RAW TELEMETRY (LAST 10)")}{_tbl(feat_hist, fc, "LIVE EXTRACTED FEATURES (LAST 10)")}
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin-bottom:1.5rem;">
    {_w("Voltage",f"{v:.2f}","V","bolt")}{_w("Current",f"{c:.2f}","A","speed")}{_w("Temperature",f"{t:.2f}","C","thermostat")}
  </div>
  <div style="display:grid;grid-template-columns:1fr 2fr;gap:1.5rem;margin-bottom:1.5rem;">
    <div style="background:radial-gradient(circle at center,#1a1500 0%,#121212 100%);border:1px solid #D4AF37;border-radius:12px;padding:1.8rem;display:flex;flex-direction:column;justify-content:center;align-items:center;">
      <p style="color:#D4AF37;font-size:0.65rem;font-weight:700;letter-spacing:2px;margin:0 0 0.6rem;">AGGREGATE HEALTH</p>
      <div style="font-family:JetBrains Mono,monospace;font-size:3.5rem;font-weight:700;color:#fff;line-height:1;">{hs}<span style="font-size:1.3rem;color:#D4AF37;">%</span></div>
      <div style="margin-top:0.6rem;padding:0.3rem 0.8rem;border-radius:30px;font-weight:700;font-size:0.7rem;letter-spacing:1.5px;color:{hc};background:{hc}15;border:1px solid {hc}44;">{h}</div>
    </div>
    <div style="display:grid;grid-template-rows:1fr 1fr;gap:1.5rem;">
      <div style="background:#121212;border:1px solid #222;border-radius:12px;padding:1.2rem;display:flex;align-items:center;justify-content:space-between;">
        <div><p style="color:#888;font-size:0.6rem;font-weight:700;letter-spacing:1.5px;margin:0 0 0.2rem;">ISOLATION FOREST</p><h3 style="color:#fff;font-size:1rem;font-weight:600;margin:0;">Anomaly Detection</h3></div>
        <div style="padding:0.4rem 0.8rem;border-radius:8px;font-weight:700;letter-spacing:1.5px;color:{ac};background:{ac}15;border:1px solid {ac}44;font-size:0.75rem;">{a}</div>
      </div>
      <div style="background:#121212;border:1px solid #222;border-radius:12px;padding:1.2rem;display:flex;align-items:center;justify-content:space-between;">
        <div><p style="color:#888;font-size:0.6rem;font-weight:700;letter-spacing:1.5px;margin:0 0 0.2rem;">LSTM SEQUENCE MODEL</p><h3 style="color:#fff;font-size:1rem;font-weight:600;margin:0;">Remaining Useful Life</h3></div>
        <div style="display:flex;align-items:baseline;gap:0.3rem;"><span style="font-family:JetBrains Mono,monospace;font-size:2.2rem;font-weight:700;color:#D4AF37;">{rul}</span><span style="color:#888;font-weight:600;font-size:0.7rem;">CYCLES</span></div>
      </div>
    </div>
  </div>
  {_fault_panel()}
</div></div>'''

# --- Hero ---
HERO = '''<div style="background:#050505;color:#fff;font-family:Inter,sans-serif;padding:3rem 2rem 2rem;">
<div style="max-width:1400px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:3rem;align-items:center;min-height:70vh;">
  <div>
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
      <div style="width:44px;height:44px;background:linear-gradient(135deg,#f2ca50,#b8860b);border-radius:12px;display:flex;align-items:center;justify-content:center;"><span class="material-symbols-outlined" style="color:#000;font-size:26px;">bolt</span></div>
      <span style="font-family:Hanken Grotesk,sans-serif;font-size:0.9rem;font-weight:700;letter-spacing:3px;color:#D4AF37;">VOLTGUARD AI</span>
    </div>
    <h1 style="font-family:Hanken Grotesk,sans-serif;font-size:2.8rem;font-weight:700;color:#fff;line-height:1.15;margin:0 0 1.2rem;">Battery Telemetry<br><span style="color:#D4AF37;">Digital Twin</span></h1>
    <p style="color:#999;font-size:0.95rem;line-height:1.7;margin-bottom:1.8rem;">A predictive maintenance system fusing real-time IoT sensor telemetry with three ML models trained on the NASA Battery Aging Dataset to forecast degradation, detect thermal anomalies, and estimate remaining useful life.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
      <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:12px;padding:0.9rem;"><div style="color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:2px;margin-bottom:0.3rem;">CLASSIFICATION</div><div style="color:#fff;font-weight:600;font-size:0.85rem;">XGBoost Health Classifier</div><div style="color:#666;font-size:0.7rem;">Healthy / Degraded / Critical</div></div>
      <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:12px;padding:0.9rem;"><div style="color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:2px;margin-bottom:0.3rem;">ANOMALY DETECTION</div><div style="color:#fff;font-weight:600;font-size:0.85rem;">Isolation Forest Engine</div><div style="color:#666;font-size:0.7rem;">Thermal runaway scoring</div></div>
      <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:12px;padding:0.9rem;"><div style="color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:2px;margin-bottom:0.3rem;">FORECASTING</div><div style="color:#fff;font-weight:600;font-size:0.85rem;">LSTM Sequence Model</div><div style="color:#666;font-size:0.7rem;">Remaining Useful Life (RUL)</div></div>
      <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:12px;padding:0.9rem;"><div style="color:#D4AF37;font-size:0.6rem;font-weight:700;letter-spacing:2px;margin-bottom:0.3rem;">FEATURE ENGINEERING</div><div style="color:#fff;font-weight:600;font-size:0.85rem;">Rolling Window DSP</div><div style="color:#666;font-size:0.7rem;">FFT, dV/dt, variance, spikes</div></div>
    </div>
    <div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">ESP32 + Wokwi</span>
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">MQTT (HiveMQ)</span>
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">TensorFlow/Keras</span>
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">XGBoost</span>
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">scikit-learn</span>
      <span style="background:#1a1500;color:#D4AF37;padding:0.3rem 0.7rem;border-radius:20px;font-size:0.65rem;font-weight:600;border:1px solid #333;">NASA Dataset</span>
    </div>
  </div>
  <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:16px;height:100%;min-height:400px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;">
    <div style="position:absolute;inset:0;background:radial-gradient(circle at 30% 40%,rgba(212,175,55,0.04) 0%,transparent 60%);"></div>
    <div style="text-align:center;z-index:1;"><span class="material-symbols-outlined" style="font-size:64px;color:#333;">developer_board</span><p style="color:#444;font-size:0.75rem;font-weight:600;letter-spacing:1.5px;margin-top:0.8rem;">PCB DIAGRAM PLACEHOLDER</p><p style="color:#333;font-size:0.65rem;">Replace with your hardware schematic</p></div>
  </div>
</div>
<div style="text-align:center;padding:1.5rem 0 0.5rem;animation:bounce 2s infinite;"><span class="material-symbols-outlined" style="color:#D4AF37;font-size:24px;">keyboard_double_arrow_down</span><p style="color:#555;font-size:0.7rem;letter-spacing:2px;font-weight:600;">SCROLL FOR LIVE DASHBOARD</p></div>
</div><style>@keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(8px)}}</style>'''

HEAD='''<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;700&family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script>function forceDark(){document.body.classList.add('dark');document.documentElement.classList.add('dark');}document.addEventListener("DOMContentLoaded",forceDark);setTimeout(forceDark,100);</script>'''

CSS="""body,.gradio-container{background:#050505!important;color:#fff!important;}footer{display:none!important;}.dark{background:#050505!important;}
.src-btn{background:linear-gradient(135deg,#f2ca50,#b8860b)!important;color:#000!important;font-weight:700!important;letter-spacing:1px!important;border:none!important;border-radius:10px!important;min-height:50px!important;}
.src-btn:hover{transform:translateY(-2px)!important;box-shadow:0 4px 15px rgba(212,175,55,0.3)!important;}
.fault-btn{background:#121212!important;color:#D4AF37!important;border:1px solid #333!important;border-radius:8px!important;font-weight:700!important;letter-spacing:1px!important;text-transform:uppercase!important;min-height:42px!important;}
.fault-btn:hover{background:#1a1500!important;border-color:#D4AF37!important;}
.overheat:hover{background:#2a0000!important;border-color:#FF3366!important;color:#FF3366!important;}
.clear:hover{background:#002a11!important;border-color:#00FFCC!important;color:#00FFCC!important;}"""

with gr.Blocks(css=CSS, head=HEAD, theme=gr.themes.Monochrome().set(body_background_fill="#050505",body_text_color="#ffffff",background_fill_primary="#050505")) as demo:
    gr.HTML(HERO)
    gr.HTML("<h3 style='color:#D4AF37;text-align:center;letter-spacing:2px;font-size:0.8rem;margin-bottom:0.3rem;'>SELECT DATA SOURCE</h3>")
    with gr.Row():
        bw=gr.Button("RUN WITH WOKWI (LIVE HARDWARE)",elem_classes="src-btn")
        bs=gr.Button("RUN WITH SAMPLE READINGS (NASA DATASET)",elem_classes="src-btn")
    live=gr.HTML(value=build())
    gr.HTML("<h3 style='color:#D4AF37;text-align:center;letter-spacing:2px;font-size:0.75rem;margin-top:0.8rem;'>FAULT INJECTION CONTROL</h3>")
    with gr.Row():
        f1=gr.Button("TRIGGER THERMAL RUNAWAY",elem_classes="fault-btn overheat")
        f2=gr.Button("INJECT VOLTAGE SAG",elem_classes="fault-btn")
        f3=gr.Button("INJECT CURRENT INSTABILITY",elem_classes="fault-btn")
        f4=gr.Button("RESTORE NORMAL OPERATION",elem_classes="fault-btn clear")
    bs.click(fn=start_sim,outputs=live)
    bw.click(fn=start_wokwi,outputs=live,js="()=>{window.open('https://wokwi.com/projects/464264997348774913','_blank')}")
    f1.click(fn=lambda:do_fault("overheat"),outputs=live)
    f2.click(fn=lambda:do_fault("voltage_sag"),outputs=live)
    f3.click(fn=lambda:do_fault("unstable"),outputs=live)
    f4.click(fn=lambda:do_fault("normal"),outputs=live)
    timer=gr.Timer(1); timer.tick(fn=tick,outputs=live)

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860)
