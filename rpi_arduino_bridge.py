#!/usr/bin/env python3
"""
rpi_arduino_bridge.py
---------------------
Real-Time AI & GPS Bridge between Arduino Uno R3 and Raspberry Pi Zero 2 W.

Architecture:
1. Arduino Uno reads hardware sensors (MPU6050, SPS30, HC-SR04, NEO-6M GPS) and calculates
   Layer 1 (Threshold Decision: ALERT, DROWSY, FATIGUED).
2. Arduino streams sensor telemetry to Raspberry Pi over USB Serial:
   TELEM:ax,ay,az,gx,gy,gz,tilt,pm1,pm25,pm4,pm10,threshState,gpsFix,lat,lon,speed,sats,navDist,navDir
3. Raspberry Pi runs Layer 2 (AI Random Forest Classifier via predict_fatigue).
4. Raspberry Pi sends AI classification & confidence back to Arduino:
   AI:state,confidence
5. Arduino fuses Layer 1 (Threshold) and Layer 2 (AI) into finalDecisionState
   and controls helmet alerts (MAX7219 matrix, Buzzer, and DFPlayer voice).

Usage on Raspberry Pi:
  # Normal operation (auto-detects Arduino on USB):
  python3 rpi_arduino_bridge.py

  # Specific serial port:
  python3 rpi_arduino_bridge.py --port /dev/ttyACM0

  # Simulation / bench test mode (without Arduino connected):
  python3 rpi_arduino_bridge.py --simulate
"""

import os
import sys
import time
import glob
import argparse
import numpy as np

# Import predict engine
from predict import predict_fatigue, load_model_and_features

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

STATE_NAMES = {0: "ALERT", 1: "DROWSY", 2: "FATIGUED"}
STATE_SYMBOLS = {
    0: "🟢 [ALERT]   ",
    1: "🟡 [DROWSY]  ",
    2: "🔴 [FATIGUED]"
}

NAV_DIRECTIONS = {
    0: "STANDBY",
    1: "STRAIGHT ↑",
    2: "RIGHT →",
    3: "LEFT ←",
    4: "REACHED 🏁",
    5: "U-TURN ⮌"
}


def find_arduino_port():
    """Auto-detects USB serial port for Arduino Uno on Linux/macOS."""
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/tty.usbmodem*",
        "/dev/cu.usbmodem*"
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None


def run_bridge(port: str, baud: int, simulate: bool):
    print("=" * 80)
    print(" SMART HELMET AI: RASPBERRY PI <-> ARDUINO SERIAL BRIDGE (+ GPS)")
    print("=" * 80)

    # 1. Load trained model
    print("[1/3] Loading Random Forest model...")
    model, feature_names = load_model_and_features()
    print(f"      Model loaded successfully ({len(feature_names)} features).")

    # 2. Connect to Serial
    ser = None
    if not simulate:
        if not SERIAL_AVAILABLE:
            print("[ERROR] pyserial is not installed. Install with: pip install pyserial", file=sys.stderr)
            sys.exit(1)

        if not port:
            port = find_arduino_port()
            if not port:
                print("[WARNING] No Arduino detected on USB serial ports.")
                print("          Switching to --simulate mode. Connect Arduino or pass --port <device>.")
                simulate = True

        if not simulate:
            try:
                print(f"[2/3] Connecting to Arduino Uno at {port} ({baud} baud)...")
                ser = serial.Serial(port, baud, timeout=1.0)
                time.sleep(2.0)  # Wait for Arduino bootloader reset
                ser.reset_input_buffer()
                print("      Serial connection established!")
            except Exception as e:
                print(f"[ERROR] Could not open serial port {port}: {e}", file=sys.stderr)
                print("        Switching to --simulate mode.")
                simulate = True

    if simulate:
        print("[2/3] Simulation mode active (synthetic telemetry generator).")

    print("[3/3] Dual-Layer AI Pipeline ACTIVE.")
    print("-" * 105)
    print(f"{'Time':<8} | {'Layer 1':<10} | {'Layer 2 (AI Model)':<22} | {'Kinematics & PM2.5':<28} | {'GPS & Navigation'}")
    print("-" * 105)

    prev_pm25 = None
    prev_pm10 = None

    try:
        while True:
            gps_fix = 0
            lat = 0.0
            lon = 0.0
            speed = 0.0
            sats = 0
            nav_dist = 0
            nav_dir = 0

            if not simulate and ser is not None:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Echo general debug / GPS messages from Arduino
                if not line.startswith("TELEM:"):
                    print(f"   [ARDUINO] {line}")
                    continue

                # Parse: TELEM:ax,ay,az,gx,gy,gz,tilt,pm1,pm25,pm4,pm10,threshState[,gpsFix,lat,lon,speed,sats,navDist,navDir]
                try:
                    payload = line[6:]
                    parts = [p.strip() for p in payload.split(",")]
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

                    # GPS fields (if present in updated telemetry)
                    if len(parts) > 12:
                        gps_fix = int(parts[12])
                    if len(parts) > 13:
                        lat = float(parts[13])
                    if len(parts) > 14:
                        lon = float(parts[14])
                    if len(parts) > 15:
                        speed = float(parts[15])
                    if len(parts) > 16:
                        sats = int(parts[16])
                    if len(parts) > 17:
                        nav_dist = int(parts[17])
                    if len(parts) > 18:
                        nav_dir = int(parts[18])

                except Exception as parse_err:
                    continue

            else:
                # Simulated telemetry loop
                time.sleep(1.0)
                sim_step = int(time.time()) % 30
                if sim_step < 12:
                    tilt = np.random.uniform(2.0, 8.0)
                    thresh_state = 0
                elif sim_step < 22:
                    tilt = np.random.uniform(19.0, 26.0)
                    thresh_state = 1
                else:
                    tilt = np.random.uniform(31.0, 38.0)
                    thresh_state = 2

                rad = np.radians(tilt)
                ax = round(-9.81 * np.sin(rad) + np.random.normal(0, 0.1), 2)
                ay = round(np.random.normal(0, 0.1), 2)
                az = round(9.81 * np.cos(rad) + np.random.normal(0, 0.15), 2)
                gx = round(np.random.normal(0, 1.2), 2)
                gy = round(np.random.normal(0, 1.0), 2)
                gz = round(np.random.normal(0, 1.5), 2)
                pm25 = round(np.random.uniform(25.0, 45.0), 1)
                pm1 = round(pm25 * 0.65, 1)
                pm4 = round(pm25 * 1.25, 1)
                pm10 = round(pm25 * 1.55, 1)

                # Simulate GPS fix
                gps_fix = 1
                lat = round(22.572646 + np.random.normal(0, 0.0001), 6)
                lon = round(88.363895 + np.random.normal(0, 0.0001), 6)
                speed = round(np.random.uniform(20.0, 35.0), 1)
                sats = 8
                nav_dist = max(0, 1200 - sim_step * 35)
                nav_dir = 1 if nav_dist > 50 else 4

            # Compute delta changes
            if prev_pm25 is None:
                pm25_change = 0.0
                pm10_change = 0.0
            else:
                pm25_change = round(pm25 - prev_pm25, 2)
                pm10_change = round(pm10 - prev_pm10, 2)
            prev_pm25 = pm25
            prev_pm10 = pm10

            # Assemble full sensor vector for Model B (15 features)
            sensor_dict = {
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz,
                "tilt_angle": tilt,
                "acc_magnitude": round(np.sqrt(ax**2 + ay**2 + az**2), 3),
                "gyro_magnitude": round(np.sqrt(gx**2 + gy**2 + gz**2), 3),
                "pm1": pm1, "pm25": pm25, "pm4": pm4, "pm10": pm10,
                "pm25_change": pm25_change, "pm10_change": pm10_change
            }

            # Run Layer 2 AI prediction
            pred_result = predict_fatigue(sensor_dict, model, feature_names)
            ai_label = pred_result["predicted_label"]
            ai_conf = pred_result["confidence"][pred_result["predicted_state"]]

            # Send AI decision back to Arduino over Serial: "AI:state,conf\n"
            if not simulate and ser is not None:
                response = f"AI:{ai_label},{ai_conf:.2f}\n"
                ser.write(response.encode("utf-8"))

            # Log to terminal
            t_str = time.strftime("%H:%M:%S")
            thresh_name = STATE_NAMES.get(thresh_state, "UNKNOWN")
            ai_sym = STATE_SYMBOLS.get(ai_label, "[UNKNOWN] ")

            # GPS string formatting
            if gps_fix:
                nav_str = NAV_DIRECTIONS.get(nav_dir, "")
                gps_str = f"Fix:OK ({sats} sats, {speed:4.1f}km/h) {lat:.4f},{lon:.4f} | {nav_str}"
            else:
                gps_str = f"Searching satellites ({sats} in view)..."

            print(f"{t_str} | {thresh_name:<10} | {ai_sym} ({ai_conf*100:4.1f}%) | "
                  f"Tilt:{tilt:4.1f}° PM2.5:{pm25:4.1f} | {gps_str}")

    except KeyboardInterrupt:
        print("\n[STOP] Stopping AI bridge.")
    finally:
        if ser is not None and ser.is_open:
            ser.close()


def main():
    parser = argparse.ArgumentParser(description="Smart Helmet Arduino <-> Raspberry Pi AI Bridge (+ GPS)")
    parser.add_argument("--port", type=str, default=None, help="Serial port for Arduino Uno (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without physical Arduino")
    args = parser.parse_args()

    run_bridge(args.port, args.baud, args.simulate)


if __name__ == "__main__":
    main()
