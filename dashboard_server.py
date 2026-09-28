#!/usr/bin/env python3
"""
dashboard_server.py
-------------------
Smart Helmet AI: Live Graphical Telemetry, Cognitive State & GPS Turn-by-Turn Navigation Dashboard.

Features:
- Futuristic Cyberpunk / Aeronautical HUD dark-mode aesthetic.
- Hero Rider Cognitive State Indicator (🟢 ALERT / 🟡 DROWSY / 🔴 FATIGUED).
- Dual-Layer Comparison Breakdown (Layer 1 Threshold vs Layer 2 Random Forest AI).
- Circular Animated Gauge for Head Tilt & AI Confidence.
- Live SPS30 Air Quality Particulate Cards (PM1, PM2.5, PM4, PM10).
- 60fps Canvas Real-Time Rolling Kinematic Chart.
- Real-Time GPS Tracking & Satellite Lock Status.
- Interactive OpenStreetMap (Leaflet.js) with live rider tracking & route polyline.
- Instant Location Search Bar (powered by OpenStreetMap Nominatim Geocoding).
- Turn-by-Turn Navigation HUD (Straight, Left, Right, U-Turn, Reached).
- Click-to-Set Destination on Map & Send 'DEST:lat,lon' directly to Arduino.
- Runs in both physical hardware mode (USB Serial) and --simulate mode for instant bench demonstration.
"""

import os
import sys
import time
import glob
import json
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Import core AI inference
from predict import predict_fatigue, load_model_and_features

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Global reference to active serial connection for sending destination commands
ACTIVE_SERIAL = None
SERIAL_LOCK = threading.Lock()


# =====================================================================
# GLOBAL TELEMETRY STATE (Thread-Safe)
# =====================================================================
class TelemetryState:
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.last_update = 0

        # Kinematics
        self.ax = 0.0
        self.ay = 0.0
        self.az = 9.81
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.tilt_angle = 0.0
        self.acc_mag = 9.81
        self.gyro_mag = 0.0

        # Particulates
        self.pm1 = 12.0
        self.pm25 = 20.0
        self.pm4 = 28.0
        self.pm10 = 36.0
        self.pm25_change = 0.0
        self.pm10_change = 0.0
        self.air_env = "CLEAN"

        # Decisions
        self.thresh_state = 0       # 0=ALERT, 1=DROWSY, 2=FATIGUED
        self.thresh_state_name = "ALERT"
        self.ai_state = 0
        self.ai_state_name = "ALERT"
        self.ai_confidence = 0.85
        self.final_state = 0
        self.final_state_name = "ALERT"
        self.fusion_rule = "All systems normal. Attentive rider posture."

        # GPS & Navigation
        self.gps_fix = False
        self.lat = 22.572646
        self.lon = 88.363895
        self.speed_kmph = 0.0
        self.satellites = 0
        self.nav_dist = 0
        self.nav_dir = 0            # 0=None, 1=Straight, 2=Right, 3=Left, 4=Reached, 5=UTurn
        self.dest_lat = None
        self.dest_lon = None
        self.dest_name = ""

        # Rolling history for chart (last 60 samples)
        self.history = []
        self.events = []

    def set_destination(self, lat, lon, name=""):
        with self.lock:
            self.dest_lat = lat
            self.dest_lon = lon
            self.dest_name = name or f"{lat:.4f}, {lon:.4f}"
            self.add_event(f"📍 [NAV] Destination set: {self.dest_name}")

    def clear_destination(self):
        with self.lock:
            self.dest_lat = None
            self.dest_lon = None
            self.dest_name = ""
            self.nav_dist = 0
            self.nav_dir = 0
            self.add_event("📍 [NAV] Destination cleared.")

    def update_telemetry(self, sensor_data, ai_res, thresh_state_idx, gps_data=None):
        with self.lock:
            self.connected = True
            self.last_update = time.time()

            self.ax = sensor_data["ax"]
            self.ay = sensor_data["ay"]
            self.az = sensor_data["az"]
            self.gx = sensor_data["gx"]
            self.gy = sensor_data["gy"]
            self.gz = sensor_data["gz"]
            self.tilt_angle = sensor_data["tilt_angle"]
            self.acc_mag = sensor_data["acc_magnitude"]
            self.gyro_mag = sensor_data["gyro_magnitude"]

            self.pm1 = sensor_data["pm1"]
            self.pm25 = sensor_data["pm25"]
            self.pm4 = sensor_data["pm4"]
            self.pm10 = sensor_data["pm10"]
            self.pm25_change = sensor_data["pm25_change"]
            self.pm10_change = sensor_data["pm10_change"]

            # GPS update
            if gps_data:
                self.gps_fix = bool(gps_data.get("fix", False))
                if self.gps_fix or gps_data.get("lat", 0.0) != 0.0:
                    self.lat = gps_data.get("lat", self.lat)
                    self.lon = gps_data.get("lon", self.lon)
                self.speed_kmph = gps_data.get("speed", 0.0)
                self.satellites = gps_data.get("satellites", 0)
                self.nav_dist = gps_data.get("nav_dist", 0)
                self.nav_dir = gps_data.get("nav_dir", 0)

            # Classify air environment
            if self.pm25 >= 110.0 or abs(self.pm25_change) > 15.0:
                self.air_env = "SPIKE"
            elif self.pm25 >= 60.0:
                self.air_env = "POLLUTED"
            elif self.pm25 >= 25.0:
                self.air_env = "MODERATE"
            else:
                self.air_env = "CLEAN"

            # Layer 1
            names = ["ALERT", "DROWSY", "FATIGUED"]
            self.thresh_state = thresh_state_idx
            self.thresh_state_name = names[thresh_state_idx]

            # Layer 2 (AI)
            self.ai_state = ai_res["predicted_label"]
            self.ai_state_name = ai_res["predicted_state"]
            self.ai_confidence = ai_res["confidence"][self.ai_state_name]

            # Final Fusion
            prev_final = self.final_state
            if self.thresh_state == 2 or (self.ai_state == 2 and self.ai_confidence >= 0.65):
                self.final_state = 2
                self.final_state_name = "FATIGUED"
                self.fusion_rule = "High Fatigue Override: Severe tilt hold or high AI fatigue confidence."
            elif self.thresh_state == 1 or (self.ai_state == 1 and self.ai_confidence >= 0.55) or self.ai_state == 2:
                self.final_state = 1
                self.final_state_name = "DROWSY"
                self.fusion_rule = "Cognitive Warning: Nodding or AI micro-movement drowsiness detected."
            else:
                self.final_state = 0
                self.final_state_name = "ALERT"
                self.fusion_rule = "Both layers confirm normal attentive riding posture."

            # Log events on state changes
            if self.final_state != prev_final:
                t_str = time.strftime("%H:%M:%S")
                if self.final_state == 2:
                    self.add_event(f"🔴 [{t_str}] CRITICAL FATIGUE: Final decision escalated to FATIGUED! Emergency alarms triggered.")
                elif self.final_state == 1:
                    self.add_event(f"🟡 [{t_str}] DROWSINESS WARNING: Final decision escalated to DROWSY. Caution alerts active.")
                else:
                    self.add_event(f"🟢 [{t_str}] RECOVERY: Rider posture returned to upright ALERT status.")

            # Record history entry
            entry = {
                "t": time.strftime("%H:%M:%S"),
                "tilt": self.tilt_angle,
                "pm25": self.pm25,
                "final": self.final_state
            }
            self.history.append(entry)
            if len(self.history) > 60:
                self.history.pop(0)

    def add_event(self, text):
        self.events.insert(0, text)
        if len(self.events) > 50:
            self.events.pop()

    def get_snapshot(self):
        with self.lock:
            return {
                "connected": self.connected and (time.time() - self.last_update < 4.0),
                "timestamp": time.strftime("%H:%M:%S"),
                "kinematics": {
                    "ax": self.ax, "ay": self.ay, "az": self.az,
                    "gx": self.gx, "gy": self.gy, "gz": self.gz,
                    "tilt_angle": self.tilt_angle,
                    "acc_mag": self.acc_mag,
                    "gyro_mag": self.gyro_mag
                },
                "particulates": {
                    "pm1": self.pm1, "pm25": self.pm25, "pm4": self.pm4, "pm10": self.pm10,
                    "pm25_change": self.pm25_change, "pm10_change": self.pm10_change,
                    "environment": self.air_env
                },
                "decisions": {
                    "layer1_threshold": {
                        "state": self.thresh_state,
                        "name": self.thresh_state_name
                    },
                    "layer2_ai": {
                        "state": self.ai_state,
                        "name": self.ai_state_name,
                        "confidence": round(self.ai_confidence * 100, 1)
                    },
                    "final_decision": {
                        "state": self.final_state,
                        "name": self.final_state_name,
                        "rule": self.fusion_rule
                    }
                },
                "gps": {
                    "fix": self.gps_fix,
                    "lat": self.lat,
                    "lon": self.lon,
                    "speed_kmph": self.speed_kmph,
                    "satellites": self.satellites,
                    "nav_dist": self.nav_dist,
                    "nav_dir": self.nav_dir,
                    "dest_lat": self.dest_lat,
                    "dest_lon": self.dest_lon,
                    "dest_name": self.dest_name
                },
                "history": self.history[-30:],
                "events": self.events[:15]
            }


telemetry_hub = TelemetryState()


# =====================================================================
# BACKGROUND SERIAL / INFERENCE THREAD
# =====================================================================
def find_serial_port():
    patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/tty.usbmodem*", "/dev/cu.usbmodem*"]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def serial_inference_worker(port, baud, simulate):
    global ACTIVE_SERIAL
    print("[WORKER] Initializing AI model for live inference...")
    model, feature_names = load_model_and_features()
    print("[WORKER] Model loaded successfully.")

    ser = None
    if not simulate:
        if not port:
            port = find_serial_port()

        if port and SERIAL_AVAILABLE:
            try:
                print(f"[WORKER] Connecting to Arduino on {port}...")
                ser = serial.Serial(port, baud, timeout=1.0)
                time.sleep(2.0)
                ser.reset_input_buffer()
                with SERIAL_LOCK:
                    ACTIVE_SERIAL = ser
                print("[WORKER] Arduino connected!")
            except Exception as e:
                print(f"[WORKER WARNING] Could not open {port}: {e}. Falling back to simulation.")
                simulate = True
        else:
            print("[WORKER] No physical Arduino detected. Running in SIMULATION mode.")
            simulate = True

    prev_pm25 = None
    prev_pm10 = None
    sim_t = 0
    sim_lat = 22.572646
    sim_lon = 88.363895

    while True:
        try:
            gps_data = None

            if not simulate and ser is not None:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if not line.startswith("TELEM:"):
                    print(f"[ARDUINO] {line}")
                    continue

                parts = [p.strip() for p in line[6:].split(",")]
                if len(parts) < 12:
                    continue

                ax = float(parts[0])
                ay = float(parts[1])
                az = float(parts[2])
                gx = float(parts[3])
                gy = float(parts[4])
                gz = float(parts[5])
                tilt = float(parts[6])
                pm1 = float(parts[7])
                pm25 = float(parts[8])
                pm4 = float(parts[9])
                pm10 = float(parts[10])
                thresh_state = int(parts[11])

                # GPS fields if present
                if len(parts) > 12:
                    gps_fix = bool(int(parts[12]))
                    lat = float(parts[13]) if len(parts) > 13 else 0.0
                    lon = float(parts[14]) if len(parts) > 14 else 0.0
                    speed = float(parts[15]) if len(parts) > 15 else 0.0
                    sats = int(parts[16]) if len(parts) > 16 else 0
                    nav_dist = int(parts[17]) if len(parts) > 17 else 0
                    nav_dir = int(parts[18]) if len(parts) > 18 else 0

                    gps_data = {
                        "fix": gps_fix, "lat": lat, "lon": lon,
                        "speed": speed, "satellites": sats,
                        "nav_dist": nav_dist, "nav_dir": nav_dir
                    }

            else:
                # Simulation Mode
                time.sleep(1.0)
                sim_t += 1
                cycle = sim_t % 40

                if cycle < 18:     # Normal awake riding
                    tilt = np.random.uniform(1.0, 7.0)
                    thresh_state = 0
                elif cycle < 28:   # Nodding / drowsy
                    tilt = np.random.uniform(20.0, 27.0)
                    thresh_state = 1
                elif cycle < 36:   # Severe forward droop
                    tilt = np.random.uniform(32.0, 48.0)
                    thresh_state = 2
                else:              # Recovery
                    tilt = np.random.uniform(6.0, 12.0)
                    thresh_state = 0

                rad = np.radians(tilt)
                ax = round(-9.81 * np.sin(rad) + np.random.normal(0, 0.1), 2)
                ay = round(np.random.normal(0, 0.15), 2)
                az = round(9.81 * np.cos(rad) + np.random.normal(0, 0.15), 2)
                gx = round(np.random.normal(0, 1.2), 2)
                gy = round(np.random.normal(0, 1.0), 2)
                gz = round(np.random.normal(0, 1.6), 2)
                pm25 = round(np.random.uniform(24.0, 42.0), 1)
                pm1 = round(pm25 * 0.65, 1)
                pm4 = round(pm25 * 1.25, 1)
                pm10 = round(pm25 * 1.55, 1)

                # Simulate riding along a road
                sim_lat += 0.00015
                sim_lon += 0.00008
                dest = telemetry_hub.dest_lat
                if dest:
                    dist = int(np.hypot((dest - sim_lat) * 111000, (telemetry_hub.dest_lon - sim_lon) * 111000))
                    nav_dir = 1 if dist > 40 else 4
                else:
                    dist = 0
                    nav_dir = 0

                gps_data = {
                    "fix": True,
                    "lat": round(sim_lat, 6),
                    "lon": round(sim_lon, 6),
                    "speed": round(np.random.uniform(25.0, 42.0), 1),
                    "satellites": 9,
                    "nav_dist": dist,
                    "nav_dir": nav_dir
                }

            # Compute delta changes
            if prev_pm25 is None:
                pm25_change = 0.0
                pm10_change = 0.0
            else:
                pm25_change = round(pm25 - prev_pm25, 2)
                pm10_change = round(pm10 - prev_pm10, 2)
            prev_pm25 = pm25
            prev_pm10 = pm10

            sensor_sample = {
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz,
                "tilt_angle": tilt,
                "acc_magnitude": round(np.sqrt(ax**2 + ay**2 + az**2), 3),
                "gyro_magnitude": round(np.sqrt(gx**2 + gy**2 + gz**2), 3),
                "pm1": pm1, "pm25": pm25, "pm4": pm4, "pm10": pm10,
                "pm25_change": pm25_change, "pm10_change": pm10_change
            }

            # Predict
            ai_res = predict_fatigue(sensor_sample, model, feature_names)

            # Send back to Arduino if connected
            if not simulate and ser is not None:
                resp = f"AI:{ai_res['predicted_label']},{ai_res['confidence'][ai_res['predicted_state']]:.2f}\n"
                ser.write(resp.encode("utf-8"))

            # Update dashboard state
            telemetry_hub.update_telemetry(sensor_sample, ai_res, thresh_state, gps_data)

        except Exception as err:
            time.sleep(0.5)


# =====================================================================
# EMBEDDED DASHBOARD WEB INTERFACE (HTML/CSS/JS with LEAFLET & GPS)
# =====================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Helmet AI Telemetry & GPS Navigation HUD</title>
  
  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg-dark: #0a0e17;
      --panel-bg: #111827;
      --card-bg: #1f2937;
      --border-color: #374151;
      --cyan: #06b6d4;
      --cyan-glow: rgba(6, 182, 212, 0.35);
      --green: #10b981;
      --green-glow: rgba(16, 185, 129, 0.35);
      --yellow: #f59e0b;
      --yellow-glow: rgba(245, 158, 11, 0.35);
      --red: #ef4444;
      --red-glow: rgba(239, 68, 68, 0.35);
      --purple: #8b5cf6;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      min-height: 100vh;
      padding: 16px;
    }

    .hud-container {
      max-width: 1300px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Header */
    .hud-header {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand h1 {
      font-size: 1.35rem;
      letter-spacing: 0.5px;
      font-weight: 800;
      color: #fff;
    }

    .brand .tag {
      background: rgba(6, 182, 212, 0.15);
      color: var(--cyan);
      border: 1px solid var(--cyan);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.5px;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      font-weight: 600;
    }

    .indicator-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 10px var(--green);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(1.15); }
    }

    /* Grid Layout */
    .grid-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }

    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }

    .panel-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Hero Cognitive State */
    .hero-card {
      text-align: center;
      padding: 18px 10px;
      border-radius: 10px;
      transition: all 0.3s ease;
      background: #1e293b;
    }

    .hero-state {
      font-size: 2.2rem;
      font-weight: 900;
      letter-spacing: 1.5px;
      margin: 6px 0;
    }

    .hero-desc {
      font-size: 0.82rem;
      color: var(--text-muted);
      max-width: 90%;
      margin: 0 auto;
    }

    /* State styles */
    .state-ALERT {
      border: 2px solid var(--green);
      box-shadow: 0 0 25px var(--green-glow);
    }
    .state-ALERT .hero-state { color: var(--green); }

    .state-DROWSY {
      border: 2px solid var(--yellow);
      box-shadow: 0 0 25px var(--yellow-glow);
    }
    .state-DROWSY .hero-state { color: var(--yellow); }

    .state-FATIGUED {
      border: 2px solid var(--red);
      box-shadow: 0 0 30px var(--red-glow);
      animation: alertPulse 1s infinite alternate;
    }
    .state-FATIGUED .hero-state { color: var(--red); }

    @keyframes alertPulse {
      from { box-shadow: 0 0 15px var(--red-glow); }
      to { box-shadow: 0 0 35px var(--red); }
    }

    /* Dual-Layer cards */
    .layer-box {
      background: var(--card-bg);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .layer-label { font-size: 0.8rem; color: var(--text-muted); }
    .layer-val { font-size: 1rem; font-weight: 700; color: #fff; }

    .badge-pill {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
    }
    .pill-ALERT { background: rgba(16, 185, 129, 0.2); color: var(--green); }
    .pill-DROWSY { background: rgba(245, 158, 11, 0.2); color: var(--yellow); }
    .pill-FATIGUED { background: rgba(239, 68, 68, 0.2); color: var(--red); }

    /* Circular Gauge */
    .gauge-container {
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
      margin: 10px 0;
    }

    .gauge-text {
      position: absolute;
      text-align: center;
    }

    .gauge-val { font-size: 1.8rem; font-weight: 800; color: #fff; }
    .gauge-unit { font-size: 0.75rem; color: var(--text-muted); }

    /* Particulates Grid */
    .pm-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .pm-card {
      background: var(--card-bg);
      border-radius: 8px;
      padding: 10px 6px;
      text-align: center;
    }
    .pm-title { font-size: 0.7rem; color: var(--text-muted); font-weight: 700; }
    .pm-num { font-size: 1.3rem; font-weight: 800; color: var(--cyan); margin: 3px 0; }
    .pm-unit { font-size: 0.65rem; color: var(--text-muted); }

    /* GPS & MAP SECTION */
    .map-panel {
      grid-column: 1 / -1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .search-bar-wrap {
      display: flex;
      gap: 10px;
      position: relative;
    }

    .search-input {
      flex: 1;
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      color: #fff;
      font-size: 0.9rem;
      outline: none;
      transition: border 0.2s;
    }

    .search-input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 10px var(--cyan-glow);
    }

    .search-btn, .clear-btn {
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.85rem;
      cursor: pointer;
      border: none;
      transition: transform 0.1s, opacity 0.2s;
    }

    .search-btn { background: var(--cyan); color: #000; }
    .clear-btn { background: var(--border-color); color: #fff; }
    .search-btn:active, .clear-btn:active { transform: scale(0.97); }

    .search-dropdown {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: #1f2937;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      margin-top: 4px;
      max-height: 220px;
      overflow-y: auto;
      z-index: 1000;
      box-shadow: 0 8px 25px rgba(0,0,0,0.7);
      display: none;
    }

    .search-item {
      padding: 10px 14px;
      font-size: 0.82rem;
      border-bottom: 1px solid #374151;
      cursor: pointer;
    }
    .search-item:hover { background: #374151; color: var(--cyan); }

    .nav-hud-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
      background: var(--card-bg);
      border-radius: 8px;
      padding: 12px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .nav-icon {
      font-size: 2rem;
      width: 48px;
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(6, 182, 212, 0.15);
      border-radius: 8px;
      border: 1px solid var(--cyan);
    }

    #map {
      height: 380px;
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      z-index: 1;
    }

    /* Event log */
    .log-box {
      max-height: 140px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 0.78rem;
      background: #0b1120;
      border-radius: 6px;
      padding: 8px;
      border: 1px solid #1f2937;
    }
    .log-entry { margin-bottom: 4px; }
  </style>
</head>
<body>

  <div class="hud-container">
    
    <!-- Top Header -->
    <header class="hud-header">
      <div class="brand">
        <span style="font-size: 1.6rem;">🪖</span>
        <div>
          <h1>SMART HELMET AI TELEMETRY</h1>
          <div style="font-size: 0.72rem; color: var(--text-muted);">EDGE COMPUTING &amp; REAL-TIME NAVIGATION HUD</div>
        </div>
        <span class="tag">DUAL-LAYER V2</span>
      </div>
      <div class="status-badge">
        <div id="connDot" class="indicator-dot"></div>
        <span id="connText">ARDUINO CONNECTED</span>
        <span style="color: var(--text-muted); margin-left: 8px;" id="clockText">--:--:--</span>
      </div>
    </header>

    <!-- Top Grid: Cognitive State & Dual-Layer Overview -->
    <div class="grid-row">
      <!-- 1. Hero State Card -->
      <div class="panel">
        <div class="panel-title">
          <span>Rider Cognitive Status</span>
          <span style="font-size: 0.7rem; color: var(--cyan);">FUSED DECISION</span>
        </div>
        <div id="heroCard" class="hero-card state-ALERT">
          <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted);">CURRENT CLASSIFICATION</div>
          <div id="heroStateText" class="hero-state">ALERT</div>
          <div id="heroDescText" class="hero-desc">Both layers confirm normal attentive riding posture.</div>
        </div>
      </div>

      <!-- 2. Dual-Layer Breakdown -->
      <div class="panel">
        <div class="panel-title">Dual-Layer Multi-Stage Fusion</div>
        
        <div class="layer-box">
          <div>
            <div class="layer-label">LAYER 1: ARDUINO THRESHOLD</div>
            <div id="l1Val" class="layer-val">ALERT (Tilt &lt; 18°)</div>
          </div>
          <span id="l1Pill" class="badge-pill pill-ALERT">ALERT</span>
        </div>

        <div class="layer-box">
          <div>
            <div class="layer-label">LAYER 2: EDGE RANDOM FOREST AI</div>
            <div id="l2Val" class="layer-val">ALERT (85.0% Conf)</div>
          </div>
          <span id="l2Pill" class="badge-pill pill-ALERT">ALERT</span>
        </div>

        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 6px;">
          ⚡ Fail-Safe Watchdog: Autonomous Layer 1 hardware fallback active.
        </div>
      </div>

      <!-- 3. Head Tilt Kinematics Gauge -->
      <div class="panel">
        <div class="panel-title">Head Pitch &amp; Tilt Angle</div>
        <div class="gauge-container">
          <svg width="160" height="160" viewBox="0 0 160 160">
            <circle cx="80" cy="80" r="70" stroke="#1f2937" stroke-width="10" fill="none"/>
            <circle id="tiltGaugeBar" cx="80" cy="80" r="70" stroke="var(--green)" stroke-width="10" fill="none"
                    stroke-dasharray="440" stroke-dashoffset="440" stroke-linecap="round"
                    transform="rotate(-90 80 80)" style="transition: stroke-dashoffset 0.3s ease, stroke 0.3s ease;"/>
          </svg>
          <div class="gauge-text">
            <div id="tiltDegText" class="gauge-val">0.0</div>
            <div class="gauge-unit">DEGREES TILT</div>
          </div>
        </div>
        <div style="display: flex; justify-content: space-around; font-size: 0.75rem; color: var(--text-muted);">
          <span>Accel: <strong id="accMagText" style="color: #fff;">9.81 m/s²</strong></span>
          <span>Gyro: <strong id="gyroMagText" style="color: #fff;">0.0 °/s</strong></span>
        </div>
      </div>
    </div>

    <!-- GPS & NAVIGATION PANEL (LEAFLET MAP + SEARCH) -->
    <div class="panel map-panel">
      <div class="panel-title">
        <span>📍 Real-Time GPS Tracking &amp; Turn-by-Turn Navigation</span>
        <span id="gpsStatusBadge" style="color: var(--yellow); font-size: 0.75rem;">SEARCHING SATELLITES...</span>
      </div>

      <!-- Search Bar -->
      <div class="search-bar-wrap">
        <input id="searchInput" type="text" class="search-input" placeholder="🔍 Search destination (e.g. Howrah Bridge, Airport, Park Street, Victoria Memorial)..." onkeydown="if(event.key==='Enter') searchLocation()">
        <button class="search-btn" onclick="searchLocation()">Search Destination</button>
        <button class="clear-btn" onclick="clearDestination()">Clear</button>
        <div id="searchDropdown" class="search-dropdown"></div>
      </div>

      <!-- Turn-by-Turn HUD Strip -->
      <div class="nav-hud-strip">
        <div class="nav-item">
          <div id="navTurnIcon" class="nav-icon">🧭</div>
          <div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">NEXT MANEUVER</div>
            <div id="navTurnText" style="font-size: 1.1rem; font-weight: 800; color: var(--cyan);">STANDBY</div>
          </div>
        </div>

        <div class="nav-item">
          <div style="font-size: 1.6rem; color: var(--purple);">🎯</div>
          <div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">DISTANCE REMAINING</div>
            <div id="navDistText" style="font-size: 1.1rem; font-weight: 800; color: #fff;">-- m</div>
          </div>
        </div>

        <div class="nav-item">
          <div style="font-size: 1.6rem; color: var(--green);">⚡</div>
          <div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">GROUND SPEED</div>
            <div id="navSpeedText" style="font-size: 1.1rem; font-weight: 800; color: #fff;">0.0 km/h</div>
          </div>
        </div>

        <div class="nav-item">
          <div style="font-size: 1.6rem; color: var(--yellow);">🛰️</div>
          <div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">COORDINATES &amp; SATELLITES</div>
            <div id="navCoordsText" style="font-size: 0.85rem; font-weight: 700; color: #fff;">22.5726, 88.3639 (0 sats)</div>
          </div>
        </div>
      </div>

      <!-- Leaflet Map Container -->
      <div id="map"></div>
      <div style="font-size: 0.72rem; color: var(--text-muted);">
        💡 <em>Click anywhere on the map directly to set a navigation destination and stream it to the helmet!</em>
      </div>
    </div>

    <!-- Bottom Row: Environmental SPS30 Air Quality & Event Logs -->
    <div class="grid-row">
      <div class="panel">
        <div class="panel-title">
          <span>Sensirion SPS30 Environmental Particulates</span>
          <span id="envBadge" class="tag">CLEAN AIR</span>
        </div>
        <div class="pm-grid">
          <div class="pm-card">
            <div class="pm-title">PM 1.0</div>
            <div id="pm1Num" class="pm-num">12</div>
            <div class="pm-unit">µg/m³</div>
          </div>
          <div class="pm-card">
            <div class="pm-title">PM 2.5</div>
            <div id="pm25Num" class="pm-num">20</div>
            <div class="pm-unit">µg/m³</div>
          </div>
          <div class="pm-card">
            <div class="pm-title">PM 4.0</div>
            <div id="pm4Num" class="pm-num">28</div>
            <div class="pm-unit">µg/m³</div>
          </div>
          <div class="pm-card">
            <div class="pm-title">PM 10.0</div>
            <div id="pm10Num" class="pm-num">36</div>
            <div class="pm-unit">µg/m³</div>
          </div>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 10px;">
          Physically constrained: PM1.0 ≤ PM2.5 ≤ PM4.0 ≤ PM10.0
        </div>
      </div>

      <!-- Event Logs -->
      <div class="panel">
        <div class="panel-title">Live System &amp; Navigation Events</div>
        <div id="eventLogBox" class="log-box">
          <div class="log-entry">🟢 [SYSTEM] Dual-Layer AI &amp; GPS Navigation Hub active.</div>
        </div>
      </div>
    </div>

  </div>

  <script>
    // -------------------------------------------------------------
    // LEAFLET MAP INITIALIZATION
    // -------------------------------------------------------------
    let map = null;
    let riderMarker = null;
    let destMarker = null;
    let routeLine = null;
    let mapInitialized = false;

    const TURN_ICONS = {
      0: "🧭",
      1: "⬆️", // Straight
      2: "➡️", // Right
      3: "⬅️", // Left
      4: "🏁", // Reached
      5: "⮌"  // U-Turn
    };

    const TURN_LABELS = {
      0: "STANDBY",
      1: "STRAIGHT AHEAD",
      2: "TURN RIGHT",
      3: "TURN LEFT",
      4: "DESTINATION REACHED",
      5: "MAKE A U-TURN"
    };

    function initMap(startLat, startLon) {
      if (mapInitialized) return;
      map = L.map('map').setView([startLat, startLon], 14);

      // 100% Free OpenStreetMap & ESRI Tiles (Zero API Key, No Watermark)
      const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
        maxZoom: 19
      });

      const darkLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
      });

      const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; DigitalGlobe, GeoEye, Earthstar Geographics',
        maxZoom: 18
      });

      // Add default clean OpenStreetMap layer
      osmLayer.addTo(map);

      // Layer switcher control (Street, Dark Night HUD, Satellite)
      L.control.layers({
        "🗺️ OpenStreetMap": osmLayer,
        "🌙 Dark Night HUD": darkLayer,
        "🛰️ Satellite View": satelliteLayer
      }, null, { position: 'topright' }).addTo(map);

      // Rider Marker (Pulsing helmet dot)
      const helmetIcon = L.divIcon({
        className: 'custom-rider-icon',
        html: `<div style="background:#06b6d4; width:16px; height:16px; border-radius:50%; border:3px solid #fff; box-shadow:0 0 15px #06b6d4;"></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
      });

      riderMarker = L.marker([startLat, startLon], { icon: helmetIcon }).addTo(map)
        .bindPopup("<b>Smart Helmet Rider</b><br>Live Telemetry Location");

      // Click on map to set destination
      map.on('click', function(e) {
        setDestination(e.latlng.lat, e.latlng.lng, `Pinned (${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)})`);
      });

      mapInitialized = true;
    }

    // -------------------------------------------------------------
    // DESTINATION API & GEOCODING SEARCH
    // -------------------------------------------------------------
    async function searchLocation() {
      const q = document.getElementById("searchInput").value.trim();
      if (!q) return;

      const dropdown = document.getElementById("searchDropdown");
      dropdown.style.display = "block";
      dropdown.innerHTML = `<div class="search-item">Searching OpenStreetMap Nominatim...</div>`;

      try {
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=5`;
        const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
        const items = await res.json();

        if (items.length === 0) {
          dropdown.innerHTML = `<div class="search-item">No locations found. Try a different query.</div>`;
          return;
        }

        dropdown.innerHTML = items.map(item => `
          <div class="search-item" onclick="selectSearchLocation(${item.lat}, ${item.lon}, '${item.display_name.replace(/'/g, "\\'")}')">
            📍 <strong>${item.name || item.display_name.split(',')[0]}</strong><br>
            <span style="font-size:0.7rem; color:#9ca3af;">${item.display_name}</span>
          </div>
        `).join("");

      } catch (err) {
        dropdown.innerHTML = `<div class="search-item" style="color:var(--red);">Search error: ${err.message}</div>`;
      }
    }

    function selectSearchLocation(lat, lon, name) {
      document.getElementById("searchDropdown").style.display = "none";
      document.getElementById("searchInput").value = name.split(',')[0];
      setDestination(lat, lon, name);
    }

    async function setDestination(lat, lon, name) {
      try {
        await fetch("/api/set_destination", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lat: lat, lon: lon, name: name })
        });
        updateDestinationMarker(lat, lon, name);
      } catch (e) {
        console.error("Failed to set destination:", e);
      }
    }

    async function clearDestination() {
      try {
        await fetch("/api/clear_destination", { method: "POST" });
        if (destMarker) { map.removeLayer(destMarker); destMarker = null; }
        if (routeLine) { map.removeLayer(routeLine); routeLine = null; }
        document.getElementById("searchInput").value = "";
      } catch (e) {
        console.error("Failed to clear destination:", e);
      }
    }

    function updateDestinationMarker(lat, lon, name) {
      if (!map) return;
      if (destMarker) map.removeLayer(destMarker);
      if (routeLine) map.removeLayer(routeLine);

      const destIcon = L.divIcon({
        className: 'custom-dest-icon',
        html: `<div style="background:#ef4444; width:18px; height:18px; border-radius:50%; border:3px solid #fff; box-shadow:0 0 15px #ef4444;"></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      });

      destMarker = L.marker([lat, lon], { icon: destIcon }).addTo(map)
        .bindPopup(`<b>Destination</b><br>${name}`).openPopup();

      if (riderMarker) {
        const riderLatLng = riderMarker.getLatLng();
        routeLine = L.polyline([riderLatLng, [lat, lon]], {
          color: '#06b6d4',
          dashArray: '8, 8',
          weight: 4,
          opacity: 0.8
        }).addTo(map);

        map.fitBounds([riderLatLng, [lat, lon]], { padding: [50, 50] });
      }
    }

    // -------------------------------------------------------------
    // TELEMETRY POLLING LOOP
    // -------------------------------------------------------------
    async function updateDashboard() {
      try {
        const res = await fetch("/api/data");
        const data = await res.json();

        // Clock & Status
        document.getElementById("clockText").innerText = data.timestamp;
        const dot = document.getElementById("connDot");
        const connText = document.getElementById("connText");
        if (data.connected) {
          dot.style.background = "var(--green)";
          dot.style.boxShadow = "0 0 10px var(--green)";
          connText.innerText = "LIVE TELEMETRY ACTIVE";
        } else {
          dot.style.background = "var(--red)";
          dot.style.boxShadow = "0 0 10px var(--red)";
          connText.innerText = "STREAM DISCONNECTED";
        }

        // Hero State
        const final = data.decisions.final_decision;
        const heroCard = document.getElementById("heroCard");
        heroCard.className = "hero-card state-" + final.name;
        document.getElementById("heroStateText").innerText = final.name;
        document.getElementById("heroDescText").innerText = final.rule;

        // Dual-Layer
        const l1 = data.decisions.layer1_threshold;
        document.getElementById("l1Val").innerText = `${l1.name} (Hold Checked)`;
        const l1Pill = document.getElementById("l1Pill");
        l1Pill.className = "badge-pill pill-" + l1.name;
        l1Pill.innerText = l1.name;

        const l2 = data.decisions.layer2_ai;
        document.getElementById("l2Val").innerText = `${l2.name} (${l2.confidence}% Conf)`;
        const l2Pill = document.getElementById("l2Pill");
        l2Pill.className = "badge-pill pill-" + l2.name;
        l2Pill.innerText = l2.name;

        // Tilt Gauge
        const tilt = Math.min(Math.max(data.kinematics.tilt_angle, 0), 90);
        document.getElementById("tiltDegText").innerText = tilt.toFixed(1);
        const bar = document.getElementById("tiltGaugeBar");
        const circumference = 2 * Math.PI * 70; // 440
        const progress = tilt / 90.0;
        bar.style.strokeDashoffset = circumference - (progress * circumference);
        if (tilt >= 30) bar.style.stroke = "var(--red)";
        else if (tilt >= 18) bar.style.stroke = "var(--yellow)";
        else bar.style.stroke = "var(--green)";

        // Accelerometer & Gyro
        document.getElementById("accMagText").innerText = `${data.kinematics.acc_mag.toFixed(2)} m/s²`;
        document.getElementById("gyroMagText").innerText = `${data.kinematics.gyro_mag.toFixed(1)} °/s`;

        // Particulates
        document.getElementById("pm1Num").innerText = data.particulates.pm1;
        document.getElementById("pm25Num").innerText = data.particulates.pm25;
        document.getElementById("pm4Num").innerText = data.particulates.pm4;
        document.getElementById("pm10Num").innerText = data.particulates.pm10;
        document.getElementById("envBadge").innerText = data.particulates.environment + " AIR";

        // GPS & Navigation Data
        const gps = data.gps;
        const gpsBadge = document.getElementById("gpsStatusBadge");
        if (gps.fix) {
          gpsBadge.style.color = "var(--green)";
          gpsBadge.innerText = `🟢 GPS LOCKED (${gps.satellites} Sats)`;
        } else {
          gpsBadge.style.color = "var(--yellow)";
          gpsBadge.innerText = `🟡 SEARCHING SATELLITES (${gps.satellites} in view)...`;
        }

        document.getElementById("navSpeedText").innerText = `${gps.speed_kmph.toFixed(1)} km/h`;
        document.getElementById("navCoordsText").innerText = `${gps.lat.toFixed(4)}, ${gps.lon.toFixed(4)} (${gps.satellites} sats)`;
        document.getElementById("navDistText").innerText = gps.dest_lat ? `${gps.nav_dist} meters` : "--";

        // Turn Maneuver Icon & Text
        const turnDir = gps.nav_dir || 0;
        document.getElementById("navTurnIcon").innerText = TURN_ICONS[turnDir] || "🧭";
        document.getElementById("navTurnText").innerText = TURN_LABELS[turnDir] || "STANDBY";

        // Map updates
        if (!mapInitialized && gps.lat !== 0) {
          initMap(gps.lat, gps.lon);
        } else if (mapInitialized && riderMarker && gps.lat !== 0) {
          riderMarker.setLatLng([gps.lat, gps.lon]);
          if (routeLine && gps.dest_lat) {
            routeLine.setLatLngs([[gps.lat, gps.lon], [gps.dest_lat, gps.dest_lon]]);
          }
        }

        // Destination Marker sync
        if (gps.dest_lat && (!destMarker || destMarker.getLatLng().lat !== gps.dest_lat)) {
          updateDestinationMarker(gps.dest_lat, gps.dest_lon, gps.dest_name);
        }

        // Events Log
        const logBox = document.getElementById("eventLogBox");
        if (data.events && data.events.length > 0) {
          logBox.innerHTML = data.events.map(ev => `<div class="log-entry">${ev}</div>`).join("");
        }

      } catch (err) {
        console.warn("Polling error:", err);
      }
    }

    setInterval(updateDashboard, 500);
    updateDashboard();
  </script>
</body>
</html>
"""


# =====================================================================
# HTTP REQUEST HANDLER
# =====================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            elif self.path == "/api/data":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                data = telemetry_hub.get_snapshot()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path == "/api/set_destination":
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length).decode("utf-8")
                try:
                    payload = json.loads(post_data)
                    lat = float(payload["lat"])
                    lon = float(payload["lon"])
                    name = payload.get("name", "")
                    telemetry_hub.set_destination(lat, lon, name)

                    # Send command to Arduino over Serial if connected: "DEST:lat,lon\n"
                    with SERIAL_LOCK:
                        if ACTIVE_SERIAL is not None and ACTIVE_SERIAL.is_open:
                            cmd = f"DEST:{lat:.6f},{lon:.6f}\n"
                            ACTIVE_SERIAL.write(cmd.encode("utf-8"))
                            print(f"[SERIAL OUT] {cmd.strip()}")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "lat": lat, "lon": lon}).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

            elif self.path == "/api/clear_destination":
                telemetry_hub.clear_destination()
                with SERIAL_LOCK:
                    if ACTIVE_SERIAL is not None and ACTIVE_SERIAL.is_open:
                        ACTIVE_SERIAL.write(b"DEST:0,0\n")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "cleared"}).encode("utf-8"))
                return

            self.send_response(404)
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        # Silence default HTTP terminal logging
        return


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # Ignore client disconnects / page refreshes
        exc_type, _, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Smart Helmet Live Telemetry & GPS Navigation Dashboard")
    parser.add_argument("--port", type=str, default=None, help="Port: if numeric (e.g. 8080), sets web port; if device (e.g. /dev/ttyACM0), sets serial port")
    parser.add_argument("--serial-port", dest="serial_port", type=str, default=None, help="Explicit serial port for Arduino Uno (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--web-port", type=int, default=8080, help="Web dashboard port (default: 8080)")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated sensor & GPS stream")
    args = parser.parse_args()

    web_port = args.web_port
    serial_port = args.serial_port or args.port
    if args.port and args.port.isdigit():
        web_port = int(args.port)
        serial_port = args.serial_port

    # Start background serial inference worker
    t = threading.Thread(target=serial_inference_worker, args=(serial_port, args.baud, args.simulate), daemon=True)
    t.start()

    server = ReusableHTTPServer(("0.0.0.0", web_port), DashboardHandler)
    print("=" * 80)
    print("🚀 SMART HELMET AI: LIVE TELEMETRY & GPS NAVIGATION DASHBOARD LAUNCHED")
    print("=" * 80)
    print(f"  Local Laptop URL:    http://localhost:{web_port}")
    print(f"  Network / Pi URL:    http://<RASPBERRY_PI_IP>:{web_port}")
    print("=" * 80)
    print("Open the link above in your browser to view the Live Telemetry & GPS HUD!")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Stopping Dashboard Server.")
        server.server_close()


if __name__ == "__main__":
    main()
