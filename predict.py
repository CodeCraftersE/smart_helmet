#!/usr/bin/env python3
"""
predict.py
----------
Inference program for AI-Integrated Smart Helmet system.

Features:
- Loads the trained Random Forest model and feature definitions.
- Provides `predict_fatigue(sensor_data)` for seamless edge integration (e.g., Raspberry Pi).
- Automatically calculates derived physics features (tilt_angle, acc_magnitude, gyro_magnitude,
  pm25_change, pm10_change) if raw sensor inputs (MPU6050 + SPS30) are provided.
- Accepts inputs via command-line arguments OR runs an interactive multi-state demonstration
  using test samples from the dataset.

Usage:
  # Demonstration mode (loads samples from CSV):
  python predict.py

  # Custom sensor input:
  python predict.py --ax 4.2 --ay 0.5 --az 8.8 --gx 1.2 --gy 0.8 --gz 1.5 --pm1 25 --pm25 35 --pm4 42 --pm10 50

  # Specific CSV row:
  python predict.py --row-idx 42
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

# Paths
DEFAULT_MODEL_PATH = os.path.join("models", "smart_helmet_random_forest.joblib")
DEFAULT_FEATURES_PATH = os.path.join("models", "feature_names.json")
DEFAULT_DATA_PATH = os.path.join("data", "smart_helmet_synthetic_10000.csv")

LABEL_MAP = {0: "ALERT", 1: "DROWSY", 2: "FATIGUED"}

# Global cache for loaded model and features
_CACHED_MODEL = None
_CACHED_FEATURES = None


def load_model_and_features(model_path: str = DEFAULT_MODEL_PATH, features_path: str = DEFAULT_FEATURES_PATH):
    """Loads and caches the model and feature names list."""
    global _CACHED_MODEL, _CACHED_FEATURES
    if _CACHED_MODEL is None or _CACHED_FEATURES is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run train_model.py first.")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at: {features_path}. Please run train_model.py first.")

        _CACHED_MODEL = joblib.load(model_path)
        with open(features_path, "r") as f:
            _CACHED_FEATURES = json.load(f)

    return _CACHED_MODEL, _CACHED_FEATURES


def calculate_derived_features(data_dict: dict) -> dict:
    """
    Computes derived kinematic and environmental change features if not already present.
    Designed for Raspberry Pi sensor loops where only raw MPU6050 and SPS30 readings are read.
    """
    out = dict(data_dict)

    # 1. Accelerometer magnitude
    ax = float(out["ax"])
    ay = float(out["ay"])
    az = float(out["az"])
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    if "acc_magnitude" not in out:
        out["acc_magnitude"] = round(acc_mag, 3)

    # 2. Tilt angle relative to vertical reference [0, 0, 1]
    if "tilt_angle" not in out:
        cos_theta = np.clip(az / max(acc_mag, 1e-6), -1.0, 1.0)
        tilt = np.degrees(np.arccos(cos_theta))
        out["tilt_angle"] = round(tilt, 2)

    # 3. Gyroscope magnitude
    gx = float(out["gx"])
    gy = float(out["gy"])
    gz = float(out["gz"])
    if "gyro_magnitude" not in out:
        out["gyro_magnitude"] = round(np.sqrt(gx**2 + gy**2 + gz**2), 3)

    # 4. Particulate changes (default to 0.0 if not tracked in single call)
    if "pm25_change" not in out:
        out["pm25_change"] = 0.0
    if "pm10_change" not in out:
        out["pm10_change"] = 0.0

    return out


def predict_fatigue(sensor_data, model=None, feature_names=None):
    """
    Real-Time Helmet Fatigue Predictor.
    
    Accepts:
      sensor_data: dict or pandas.DataFrame containing the sensor readings.
                   Raw readings required:
                     MPU6050: ax, ay, az, gx, gy, gz
                     SPS30:   pm1, pm25, pm4, pm10
                   Optional derived:
                     tilt_angle, acc_magnitude, gyro_magnitude, pm25_change, pm10_change
    
    Returns:
      dict with:
        'predicted_label': 0, 1, or 2
        'predicted_state': 'ALERT', 'DROWSY', or 'FATIGUED'
        'confidence': {'ALERT': p0, 'DROWSY': p1, 'FATIGUED': p2}
        'features_used': dict of all 15 features
    """
    if model is None or feature_names is None:
        model, feature_names = load_model_and_features()

    is_dict = isinstance(sensor_data, dict)
    if is_dict:
        processed_data = calculate_derived_features(sensor_data)
        df_input = pd.DataFrame([processed_data])
    else:
        df_input = sensor_data.copy()
        for idx in df_input.index:
            row_dict = df_input.loc[idx].to_dict()
            processed = calculate_derived_features(row_dict)
            for k, v in processed.items():
                df_input.loc[idx, k] = v

    # Verify all model features are present
    missing = [f for f in feature_names if f not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required sensor features: {missing}")

    X = df_input[feature_names]

    # Predict label & probabilities
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for i in range(len(X)):
        pred_label = int(preds[i])
        pred_state = LABEL_MAP.get(pred_label, "UNKNOWN")
        conf_dict = {}
        if probs is not None:
            for cls_idx, cls_name in LABEL_MAP.items():
                # Check if class exists in model.classes_
                if cls_idx in model.classes_:
                    col_idx = list(model.classes_).index(cls_idx)
                    conf_dict[cls_name] = round(float(probs[i, col_idx]), 4)
                else:
                    conf_dict[cls_name] = 0.0

        results.append({
            "predicted_label": pred_label,
            "predicted_state": pred_state,
            "confidence": conf_dict,
            "features_used": X.iloc[i].to_dict()
        })

    return results[0] if is_dict else results


def format_prediction_output(pred_result: dict, title: str = ""):
    """Prints output matching the requested specification."""
    if title:
        print(f"\n--- {title} ---")
    print(f"Predicted State: {pred_result['predicted_state']}")
    print("Confidence:")
    for state, prob in pred_result["confidence"].items():
        print(f"  {state:8s}: {prob:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Helmet Fatigue State Predictor")

    # MPU6050 Accelerometer
    parser.add_argument("--ax", type=float, default=None, help="MPU6050 X Acceleration (m/s^2)")
    parser.add_argument("--ay", type=float, default=None, help="MPU6050 Y Acceleration (m/s^2)")
    parser.add_argument("--az", type=float, default=None, help="MPU6050 Z Acceleration (m/s^2)")

    # MPU6050 Gyroscope
    parser.add_argument("--gx", type=float, default=None, help="MPU6050 X Angular Velocity (deg/s)")
    parser.add_argument("--gy", type=float, default=None, help="MPU6050 Y Angular Velocity (deg/s)")
    parser.add_argument("--gz", type=float, default=None, help="MPU6050 Z Angular Velocity (deg/s)")

    # SPS30 Particulates
    parser.add_argument("--pm1", type=float, default=None, help="SPS30 PM1 (ug/m^3)")
    parser.add_argument("--pm25", type=float, default=None, help="SPS30 PM2.5 (ug/m^3)")
    parser.add_argument("--pm4", type=float, default=None, help="SPS30 PM4 (ug/m^3)")
    parser.add_argument("--pm10", type=float, default=None, help="SPS30 PM10 (ug/m^3)")

    # Optional precomputed features
    parser.add_argument("--tilt-angle", type=float, default=None, help="Head tilt angle in degrees")
    parser.add_argument("--acc-magnitude", type=float, default=None, help="Total acceleration magnitude")
    parser.add_argument("--gyro-magnitude", type=float, default=None, help="Total angular velocity magnitude")
    parser.add_argument("--pm25-change", type=float, default=None, help="PM2.5 delta from previous second")
    parser.add_argument("--pm10-change", type=float, default=None, help="PM10 delta from previous second")

    # CSV inspection mode
    parser.add_argument("--csv", type=str, default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--row-idx", type=int, default=None, help="Evaluate a specific row index from the dataset")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and feature definition
    try:
        model, feature_names = load_model_and_features()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if manual sensor values were provided via CLI
    has_cli_motion = args.ax is not None and args.ay is not None and args.az is not None
    has_cli_sps = args.pm25 is not None

    if has_cli_motion and has_cli_sps:
        sensor_dict = {
            "ax": args.ax,
            "ay": args.ay,
            "az": args.az,
            "gx": args.gx if args.gx is not None else 0.0,
            "gy": args.gy if args.gy is not None else 0.0,
            "gz": args.gz if args.gz is not None else 0.0,
            "pm1": args.pm1 if args.pm1 is not None else args.pm25 * 0.65,
            "pm25": args.pm25,
            "pm4": args.pm4 if args.pm4 is not None else args.pm25 * 1.25,
            "pm10": args.pm10 if args.pm10 is not None else args.pm25 * 1.6,
        }
        if args.tilt_angle is not None:
            sensor_dict["tilt_angle"] = args.tilt_angle
        if args.acc_magnitude is not None:
            sensor_dict["acc_magnitude"] = args.acc_magnitude
        if args.gyro_magnitude is not None:
            sensor_dict["gyro_magnitude"] = args.gyro_magnitude
        if args.pm25_change is not None:
            sensor_dict["pm25_change"] = args.pm25_change
        if args.pm10_change is not None:
            sensor_dict["pm10_change"] = args.pm10_change

        print("=" * 60)
        print("INFERENCE ON COMMAND-LINE INPUT")
        print("=" * 60)
        print(f"Input Sensor Vector:")
        for k, v in sensor_dict.items():
            print(f"  {k:15s}: {v}")

        res = predict_fatigue(sensor_dict, model, feature_names)
        format_prediction_output(res, "Prediction Result")
        print("=" * 60)
        return

    # Check if specific row index from CSV was requested
    if args.row_idx is not None:
        if not os.path.exists(args.csv):
            print(f"Error: Dataset not found at {args.csv}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(args.csv)
        if args.row_idx < 0 or args.row_idx >= len(df):
            print(f"Error: Row index {args.row_idx} is out of bounds [0, {len(df)-1}].", file=sys.stderr)
            sys.exit(1)

        row = df.iloc[args.row_idx].to_dict()
        print("=" * 60)
        print(f"INFERENCE ON DATASET ROW {args.row_idx}")
        print("=" * 60)
        print(f"Session: {row.get('session_id')}, Subject: {row.get('subject_id')}, Timestamp: {row.get('timestamp')}")
        print(f"Ground Truth Label: {row.get('label_name')} ({row.get('label')}) | Environment: {row.get('environment')}")
        print(f"Sensor Values: ax={row.get('ax')}, ay={row.get('ay')}, az={row.get('az')}, tilt={row.get('tilt_angle')} deg, PM2.5={row.get('pm25')}")

        res = predict_fatigue(row, model, feature_names)
        format_prediction_output(res, "Prediction Result")
        print("=" * 60)
        return

    # Default demonstration mode:
    # Pick 3 representative samples (one ALERT, one DROWSY, one FATIGUED) from the CSV dataset
    print("=" * 60)
    print("SMART HELMET FATIGUE PREDICTOR — DEMONSTRATION MODE")
    print("=" * 60)

    if not os.path.exists(args.csv):
        print(f"Error: Dataset {args.csv} not found. Please run generate_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # Pick samples from test sessions (e.g. SESS_11, SESS_07, SESS_08)
    samples_to_show = []
    for target_label, label_name in [(0, "ALERT"), (1, "DROWSY"), (2, "FATIGUED")]:
        match = df[df["label"] == target_label]
        if not match.empty:
            # Pick a sample from a test session
            sample_row = match.iloc[len(match) // 3].to_dict()
            samples_to_show.append((label_name, sample_row))

    for target_name, row in samples_to_show:
        print(f"\nEvaluating Sample with Ground Truth: {target_name} (Session: {row['session_id']}, Env: {row['environment']})")
        print(f"  Input readings: ax={row['ax']:5.2f}, ay={row['ay']:5.2f}, az={row['az']:5.2f}, "
              f"tilt={row['tilt_angle']:5.1f} deg, gyro_mag={row['gyro_magnitude']:5.2f} deg/s, "
              f"PM2.5={row['pm25']:5.1f} ug/m3")

        # Pass raw readings without pre-computed derived features to test automatic feature calculation
        raw_input = {
            "ax": row["ax"],
            "ay": row["ay"],
            "az": row["az"],
            "gx": row["gx"],
            "gy": row["gy"],
            "gz": row["gz"],
            "pm1": row["pm1"],
            "pm25": row["pm25"],
            "pm4": row["pm4"],
            "pm10": row["pm10"],
            "pm25_change": row["pm25_change"],
            "pm10_change": row["pm10_change"],
        }
        res = predict_fatigue(raw_input, model, feature_names)
        format_prediction_output(res)

    print("\n" + "=" * 60)
    print("DEMONSTRATION OF REAL-TIME HELMET COMPATIBILITY:")
    print("Simulating real-time sensor loop from Raspberry Pi...")
    rpi_sample = {
        "ax": 3.85, "ay": 0.42, "az": 8.95,
        "gx": 1.20, "gy": 0.55, "gz": 0.80,
        "pm1": 22.4, "pm25": 34.8, "pm4": 42.1, "pm10": 53.6
    }
    print(f"Raw Pi reading: {rpi_sample}")
    res = predict_fatigue(rpi_sample, model, feature_names)
    format_prediction_output(res, "Pi Sensor Prediction")
    print("=" * 60)


if __name__ == "__main__":
    main()
