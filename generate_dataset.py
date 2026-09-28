#!/usr/bin/env python3
"""
generate_dataset.py
-------------------
Synthetic Dataset Generation for AI-Integrated Smart Helmet System.

Generates 10,000 rows of realistic time-series sensor data simulating:
1. MPU6050: 6-DoF accelerometer (ax, ay, az) and gyroscope (gx, gy, gz)
2. SPS30: Optical particulate matter sensor (PM1, PM2.5, PM4, PM10)
3. Derived features: tilt_angle, acc_magnitude, gyro_magnitude, pm25_change, pm10_change
4. Context & Labels: environment, fatigue_score, label, label_name

Classes:
  0 -> ALERT
  1 -> DROWSY
  2 -> FATIGUED

The SPS30 sensor acts as environmental/contextual exposure data, NOT a direct fatigue detector.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "smart_helmet_synthetic_10000.csv")

# Constants
GRAVITY = 9.81  # m/s^2
TOTAL_ROWS = 10000
NUM_SUBJECTS = 20
SESSIONS_PER_SUBJECT = 2
TOTAL_SESSIONS = NUM_SUBJECTS * SESSIONS_PER_SUBJECT  # 40 sessions
ROWS_PER_SESSION = TOTAL_ROWS // TOTAL_SESSIONS       # 250 rows per session


class SubjectProfile:
    """Subject-specific physiological and behavioral characteristics."""
    def __init__(self, subject_id: int):
        self.subject_id = f"SUB_{subject_id:02d}"
        # Baseline neutral head posture offsets (degrees)
        self.base_pitch_offset = np.random.uniform(-4.0, 4.0)
        self.base_roll_offset = np.random.uniform(-3.0, 3.0)
        # Movement and tremor scale factors
        self.movement_scale = np.random.uniform(0.8, 1.25)
        self.gyro_noise_scale = np.random.uniform(0.85, 1.2)
        # Nodding recovery sluggishness multiplier
        self.recovery_sluggishness = np.random.uniform(0.85, 1.3)
        # Particulate calibration / micro-location variance (±4%)
        self.pm_bias = np.random.uniform(0.96, 1.04)


def generate_session_data(subject: SubjectProfile, session_idx: int, session_type: str, env_type: str):
    """
    Generates time-series sensor data for a single riding session (250 steps, ~1 Hz).
    """
    session_id = f"SESS_{session_idx:02d}"
    dt = 1.0  # 1 second sampling interval
    start_time = datetime(2026, 3, 15, 8, 0, 0) + timedelta(minutes=session_idx * 15)

    # -------------------------------------------------------------
    # 1. Environmental Simulation (SPS30)
    # -------------------------------------------------------------
    # Define baseline PM2.5 target level based on assigned environmental condition
    if env_type == "CLEAN":
        base_pm25_target = np.random.uniform(8.0, 18.0)
    elif env_type == "MODERATE":
        base_pm25_target = np.random.uniform(32.0, 48.0)
    elif env_type == "POLLUTED":
        base_pm25_target = np.random.uniform(70.0, 100.0)
    else:  # SPIKE
        base_pm25_target = np.random.uniform(60.0, 85.0)

    # Autoregressive PM level generation
    pm25_series = np.zeros(ROWS_PER_SESSION)
    curr_pm25 = base_pm25_target * subject.pm_bias

    # Inject transient spikes (e.g., following a diesel truck or construction dust)
    spike_centers = []
    if env_type == "SPIKE":
        # Guaranteed major spike
        spike_centers.append((np.random.randint(60, 180), np.random.uniform(70.0, 120.0), np.random.uniform(15.0, 30.0)))
    elif np.random.rand() < 0.25:  # Occasional brief spike in other environments
        spike_centers.append((np.random.randint(70, 180), np.random.uniform(30.0, 60.0), np.random.uniform(10.0, 20.0)))

    for t in range(ROWS_PER_SESSION):
        # AR(1) process towards baseline
        noise = np.random.normal(0.0, 0.8)
        curr_pm25 = 0.95 * curr_pm25 + 0.05 * (base_pm25_target * subject.pm_bias) + noise
        curr_pm25 = max(3.0, curr_pm25)

        # Add smooth gaussian spike profile if active
        total_spike = 0.0
        for center_t, amplitude, width in spike_centers:
            total_spike += amplitude * np.exp(-0.5 * ((t - center_t) / width) ** 2)

        pm25_series[t] = curr_pm25 + total_spike

    # Derive PM1, PM4, PM10 maintaining physical constraint: PM1 <= PM2.5 <= PM4 <= PM10
    pm1_series = np.zeros(ROWS_PER_SESSION)
    pm4_series = np.zeros(ROWS_PER_SESSION)
    pm10_series = np.zeros(ROWS_PER_SESSION)

    for t in range(ROWS_PER_SESSION):
        p25 = pm25_series[t]
        # Fine subfraction ratio for PM1 (typically 0.55 - 0.75 of PM2.5)
        r_pm1 = np.clip(np.random.normal(0.65, 0.03), 0.50, 0.82)
        pm1 = p25 * r_pm1

        # Intermediate coarse fraction for PM4 (typically PM2.5 + 15% - 35%)
        delta_pm4 = p25 * np.clip(np.random.normal(0.25, 0.03), 0.12, 0.40)
        pm4 = p25 + delta_pm4

        # Coarse fraction for PM10 (typically PM4 + 20% - 50%)
        delta_pm10 = pm4 * np.clip(np.random.normal(0.35, 0.04), 0.18, 0.55)
        pm10 = pm4 + delta_pm10

        pm1_series[t] = max(1.0, pm1)
        pm4_series[t] = max(pm1_series[t] + 0.5, pm4)
        pm10_series[t] = max(pm4_series[t] + 0.5, pm10)

    # -------------------------------------------------------------
    # 2. Behavioral & Fatigue Progression Simulation
    # -------------------------------------------------------------
    # Session fatigue score trajectory (between 0.0 and 1.0)
    fatigue_scores = np.zeros(ROWS_PER_SESSION)

    if session_type == "ALERT":
        # Alert session: wanders between 0.04 and 0.28
        curr_score = np.random.uniform(0.06, 0.16)
        for t in range(ROWS_PER_SESSION):
            curr_score = 0.96 * curr_score + 0.04 * np.random.uniform(0.08, 0.22) + np.random.normal(0, 0.012)
            fatigue_scores[t] = np.clip(curr_score, 0.02, 0.32)

    elif session_type == "PROGRESSIVE":
        # Starts alert, drifts to drowsy, reaches fatigued towards end
        for t in range(ROWS_PER_SESSION):
            progress = t / ROWS_PER_SESSION
            base = 0.12 + 0.72 * (progress ** 1.3)
            fatigue_scores[t] = np.clip(base + np.random.normal(0, 0.03), 0.05, 0.95)

    elif session_type == "DROWSY":
        # Primarily drowsy: wanders in 0.30 - 0.65
        curr_score = np.random.uniform(0.35, 0.50)
        for t in range(ROWS_PER_SESSION):
            curr_score = 0.95 * curr_score + 0.05 * np.random.uniform(0.38, 0.58) + np.random.normal(0, 0.018)
            fatigue_scores[t] = np.clip(curr_score, 0.25, 0.68)

    else:  # FATIGUED
        # Heavily fatigued: wanders in 0.60 - 0.96
        curr_score = np.random.uniform(0.68, 0.82)
        for t in range(ROWS_PER_SESSION):
            curr_score = 0.95 * curr_score + 0.05 * np.random.uniform(0.65, 0.88) + np.random.normal(0, 0.02)
            fatigue_scores[t] = np.clip(curr_score, 0.55, 0.98)

    # Determine class label with realistic overlap
    labels = np.zeros(ROWS_PER_SESSION, dtype=int)
    for t in range(ROWS_PER_SESSION):
        fs = fatigue_scores[t]
        # Realistic fuzzy boundaries to ensure natural overlap
        if fs < 0.26:
            labels[t] = 0  # ALERT
        elif fs < 0.34:
            # Overlap zone between ALERT and DROWSY
            p_drowsy = (fs - 0.26) / (0.34 - 0.26)
            labels[t] = 1 if np.random.rand() < p_drowsy else 0
        elif fs < 0.58:
            labels[t] = 1  # DROWSY
        elif fs < 0.70:
            # Overlap zone between DROWSY and FATIGUED
            p_fatigued = (fs - 0.58) / (0.70 - 0.58)
            labels[t] = 2 if np.random.rand() < p_fatigued else 1
        else:
            labels[t] = 2  # FATIGUED

    # -------------------------------------------------------------
    # 3. MPU6050 Motion Dynamics (Pitch, Roll, Yaw, Gyro, Acc)
    # -------------------------------------------------------------
    # Pitch: head tilt forward (positive) / backward (negative)
    # Roll: head tilt right (positive) / left (negative)
    # Yaw: head turn right (positive) / left (negative)
    pitch_angles = np.zeros(ROWS_PER_SESSION)
    roll_angles = np.zeros(ROWS_PER_SESSION)
    gx_series = np.zeros(ROWS_PER_SESSION)
    gy_series = np.zeros(ROWS_PER_SESSION)
    gz_series = np.zeros(ROWS_PER_SESSION)

    curr_pitch = subject.base_pitch_offset
    curr_roll = subject.base_roll_offset
    nod_state = 0  # 0: normal, >0: active nod countdown

    for t in range(ROWS_PER_SESSION):
        lbl = labels[t]
        fs = fatigue_scores[t]

        # Behavioral dynamics conditioned on state
        if lbl == 0:  # ALERT
            # Frequent normal head movements, looking around, regular mirror checks
            target_pitch = subject.base_pitch_offset + np.random.normal(0.0, 3.0)
            target_roll = subject.base_roll_offset + np.random.normal(0.0, 2.5)

            # Occasional conscious glance down at speedo/navigation (creates overlap with higher tilt!)
            if np.random.rand() < 0.04:
                target_pitch += np.random.uniform(16.0, 26.0)

            # Fast natural recovery & active orientation
            curr_pitch = 0.70 * curr_pitch + 0.30 * target_pitch + np.random.normal(0, 0.8) * subject.movement_scale
            curr_roll = 0.70 * curr_roll + 0.30 * target_roll + np.random.normal(0, 0.7) * subject.movement_scale

            # Active gyro activity (shoulder checks, mirror scanning)
            gx = (curr_pitch - pitch_angles[max(0, t - 1)]) / dt + np.random.normal(0, 2.2) * subject.gyro_noise_scale
            gy = (curr_roll - roll_angles[max(0, t - 1)]) / dt + np.random.normal(0, 2.0) * subject.gyro_noise_scale
            gz = np.random.normal(0, 3.5) * subject.gyro_noise_scale
            if np.random.rand() < 0.08:
                # Active shoulder/mirror check
                gz += np.random.choice([-1, 1]) * np.random.uniform(18.0, 42.0)

        elif lbl == 1:  # DROWSY
            # Reduced movements during hypovigilance, interspersed with episodic head nods
            if nod_state == 0 and np.random.rand() < 0.07:
                # Initiate a nodding episode
                nod_state = np.random.randint(4, 9)

            if nod_state > 0:
                # Head dropping forward
                target_pitch = subject.base_pitch_offset + np.random.uniform(18.0, 34.0)
                curr_pitch = 0.60 * curr_pitch + 0.40 * target_pitch + np.random.normal(0, 1.2)
                nod_state -= 1
                if nod_state == 1:
                    # Abrupt head jerk / startle recovery
                    target_pitch = subject.base_pitch_offset + np.random.normal(0, 3.0)
                    gx_jerk = np.random.choice([-1, 1]) * np.random.uniform(35.0, 75.0)
                else:
                    gx_jerk = 0.0
            else:
                # Staring straight ahead with suppressed micro-movements
                target_pitch = subject.base_pitch_offset + np.random.normal(4.0, 2.5)
                curr_pitch = 0.90 * curr_pitch + 0.10 * target_pitch + np.random.normal(0, 0.4)
                gx_jerk = 0.0

            curr_roll = 0.85 * curr_roll + 0.15 * subject.base_roll_offset + np.random.normal(0, 0.6)
            gx = (curr_pitch - pitch_angles[max(0, t - 1)]) / dt + gx_jerk + np.random.normal(0, 1.4) * subject.gyro_noise_scale
            gy = (curr_roll - roll_angles[max(0, t - 1)]) / dt + np.random.normal(0, 1.2) * subject.gyro_noise_scale
            gz = np.random.normal(0, 1.6) * subject.gyro_noise_scale

        else:  # FATIGUED
            # Prolonged forward droop, very slow recovery, irregular movement
            # Notice: occasionally rider straightens up briefly (creates low-tilt fatigued samples!)
            if np.random.rand() < 0.12:
                # Brief transient upright posture
                target_pitch = subject.base_pitch_offset + np.random.normal(2.0, 3.0)
            else:
                # Persistent forward/lateral droop
                target_pitch = subject.base_pitch_offset + np.random.uniform(22.0, 42.0)

            target_roll = subject.base_roll_offset + np.random.choice([-1, 1]) * np.random.uniform(4.0, 14.0)

            # Sluggish recovery dynamics
            sluggish_alpha = 0.15 / subject.recovery_sluggishness
            curr_pitch = (1.0 - sluggish_alpha) * curr_pitch + sluggish_alpha * target_pitch + np.random.normal(0, 0.9)
            curr_roll = (1.0 - sluggish_alpha) * curr_roll + sluggish_alpha * target_roll + np.random.normal(0, 0.8)

            gx = (curr_pitch - pitch_angles[max(0, t - 1)]) / dt + np.random.normal(0, 1.8) * subject.gyro_noise_scale
            gy = (curr_roll - roll_angles[max(0, t - 1)]) / dt + np.random.normal(0, 1.6) * subject.gyro_noise_scale
            gz = np.random.normal(0, 2.0) * subject.gyro_noise_scale

        pitch_angles[t] = curr_pitch
        roll_angles[t] = curr_roll
        gx_series[t] = gx
        gy_series[t] = gy
        gz_series[t] = gz

    # -------------------------------------------------------------
    # 4. Synthesize Physical Accelerometer Measurements (ax, ay, az)
    # -------------------------------------------------------------
    ax_series = np.zeros(ROWS_PER_SESSION)
    ay_series = np.zeros(ROWS_PER_SESSION)
    az_series = np.zeros(ROWS_PER_SESSION)

    for t in range(ROWS_PER_SESSION):
        # Convert pitch and roll to radians
        rad_pitch = np.radians(pitch_angles[t])
        rad_roll = np.radians(roll_angles[t])

        # Gravitational component projected onto helmet IMU axes
        # Neutral upright: [0, 0, 1] * 9.81 m/s^2
        # Pitching forward (head down) causes negative ax and decreased az
        # Rolling right causes positive ay
        g_ax = -GRAVITY * np.sin(rad_pitch)
        g_ay = GRAVITY * np.sin(rad_roll) * np.cos(rad_pitch)
        g_az = GRAVITY * np.cos(rad_roll) * np.cos(rad_pitch)

        # Dynamic riding acceleration (braking, turning, bumps) + sensor noise
        road_vib_x = np.random.normal(0.0, 0.18)
        road_vib_y = np.random.normal(0.0, 0.18)
        road_vib_z = np.random.normal(0.0, 0.22)

        ax_series[t] = g_ax + road_vib_x
        ay_series[t] = g_ay + road_vib_y
        az_series[t] = g_az + road_vib_z

    # -------------------------------------------------------------
    # 5. Calculate Physics-Based Derived Features
    # -------------------------------------------------------------
    # acc_magnitude = sqrt(ax^2 + ay^2 + az^2)
    acc_magnitude = np.sqrt(ax_series**2 + ay_series**2 + az_series**2)

    # tilt_angle = angle between acceleration vector and upright reference [0, 0, 1]
    # cos(theta) = az / acc_magnitude
    cos_theta = np.clip(az_series / acc_magnitude, -1.0, 1.0)
    tilt_angles = np.degrees(np.arccos(cos_theta))

    # gyro_magnitude = sqrt(gx^2 + gy^2 + gz^2)
    gyro_magnitude = np.sqrt(gx_series**2 + gy_series**2 + gz_series**2)

    # pm25_change and pm10_change from previous time step (0.0 for first row of session)
    pm25_change = np.zeros(ROWS_PER_SESSION)
    pm10_change = np.zeros(ROWS_PER_SESSION)
    pm25_change[1:] = np.diff(pm25_series)
    pm10_change[1:] = np.diff(pm10_series)

    # -------------------------------------------------------------
    # 6. Environmental Context Categorization
    # -------------------------------------------------------------
    # Categorize into CLEAN, MODERATE, POLLUTED, SPIKE based on PM level & spikes
    environments = []
    for t in range(ROWS_PER_SESSION):
        p25 = pm25_series[t]
        c25 = abs(pm25_change[t])
        if p25 >= 110.0 or (c25 > 14.0 and p25 > 50.0):
            environments.append("SPIKE")
        elif p25 >= 60.0:
            environments.append("POLLUTED")
        elif p25 >= 25.0:
            environments.append("MODERATE")
        else:
            environments.append("CLEAN")

    # Timestamps
    timestamps = [(start_time + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") for t in range(ROWS_PER_SESSION)]

    # Label names
    label_map = {0: "ALERT", 1: "DROWSY", 2: "FATIGUED"}
    label_names = [label_map[lbl] for lbl in labels]

    # Assemble session DataFrame
    session_df = pd.DataFrame({
        "session_id": session_id,
        "subject_id": subject.subject_id,
        "timestamp": timestamps,
        "ax": np.round(ax_series, 3),
        "ay": np.round(ay_series, 3),
        "az": np.round(az_series, 3),
        "gx": np.round(gx_series, 3),
        "gy": np.round(gy_series, 3),
        "gz": np.round(gz_series, 3),
        "tilt_angle": np.round(tilt_angles, 2),
        "acc_magnitude": np.round(acc_magnitude, 3),
        "gyro_magnitude": np.round(gyro_magnitude, 3),
        "pm1": np.round(pm1_series, 2),
        "pm25": np.round(pm25_series, 2),
        "pm4": np.round(pm4_series, 2),
        "pm10": np.round(pm10_series, 2),
        "pm25_change": np.round(pm25_change, 2),
        "pm10_change": np.round(pm10_change, 2),
        "environment": environments,
        "fatigue_score": np.round(fatigue_scores, 4),
        "label": labels,
        "label_name": label_names,
    })

    return session_df


def main():
    print("=" * 60)
    print("SMART HELMET SYNTHETIC DATASET GENERATOR")
    print(f"Random Seed: {SEED}")
    print(f"Target Rows: {TOTAL_ROWS}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize 20 distinct subjects
    subjects = [SubjectProfile(i + 1) for i in range(NUM_SUBJECTS)]

    # Plan balanced environmental conditions and behavioral progressions
    # 40 sessions total
    session_types = (
        ["ALERT"] * 12 +
        ["PROGRESSIVE"] * 14 +
        ["DROWSY"] * 8 +
        ["FATIGUED"] * 6
    )
    # Shuffle with fixed seed for balanced assignment to subjects
    np.random.seed(SEED + 101)
    np.random.shuffle(session_types)

    env_conditions = (
        ["CLEAN"] * 11 +
        ["MODERATE"] * 12 +
        ["POLLUTED"] * 11 +
        ["SPIKE"] * 6
    )
    np.random.seed(SEED + 202)
    np.random.shuffle(env_conditions)

    # Reset seed for reproducible execution
    np.random.seed(SEED)
    random.seed(SEED)

    session_dfs = []
    session_idx = 1
    for subj in subjects:
        for _ in range(SESSIONS_PER_SUBJECT):
            st = session_types[session_idx - 1]
            et = env_conditions[session_idx - 1]
            sdf = generate_session_data(subj, session_idx, st, et)
            session_dfs.append(sdf)
            session_idx += 1

    # Concatenate all sessions
    full_df = pd.concat(session_dfs, ignore_index=True)

    # -------------------------------------------------------------
    # DATA QUALITY CHECKS & VERIFICATIONS
    # -------------------------------------------------------------
    print("\nRunning automated data quality verifications...")

    # Check 1: Exactly 10,000 rows
    assert len(full_df) == TOTAL_ROWS, f"Row count mismatch: expected {TOTAL_ROWS}, got {len(full_df)}"
    print(f" [PASS] Total rows: {len(full_df)}")

    # Check 2: No duplicate rows
    duplicates = full_df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows!"
    print(f" [PASS] Duplicate rows: {duplicates}")

    # Check 3: No NaN / null values
    nans = full_df.isna().sum().sum()
    assert nans == 0, f"Found {nans} NaN values!"
    print(f" [PASS] Missing / NaN values: {nans}")

    # Check 4: PM1 <= PM25 <= PM4 <= PM10
    pm_order_violations = (
        (full_df["pm1"] > full_df["pm25"] + 1e-4) |
        (full_df["pm25"] > full_df["pm4"] + 1e-4) |
        (full_df["pm4"] > full_df["pm10"] + 1e-4)
    ).sum()
    assert pm_order_violations == 0, f"Found {pm_order_violations} PM ordering violations!"
    print(f" [PASS] PM order constraint (PM1 <= PM25 <= PM4 <= PM10): 100% compliant (0 violations)")

    # Check 5: Accelerometer magnitude is physically reasonable
    min_acc = full_df["acc_magnitude"].min()
    max_acc = full_df["acc_magnitude"].max()
    mean_acc = full_df["acc_magnitude"].mean()
    assert 7.0 <= min_acc and max_acc <= 14.0, f"Unreasonable acc magnitude range: [{min_acc}, {max_acc}]"
    print(f" [PASS] Acc magnitude reasonable: min={min_acc:.2f}, max={max_acc:.2f}, mean={mean_acc:.2f} m/s^2")

    # Check 6: Gyroscope values reasonable
    max_gyro = full_df["gyro_magnitude"].max()
    assert max_gyro < 350.0, f"Absurd gyroscope magnitude detected: {max_gyro}"
    print(f" [PASS] Gyro magnitude reasonable: max={max_gyro:.2f} deg/s")

    # Check 7: All 3 fatigue classes exist
    unique_labels = sorted(full_df["label"].unique().tolist())
    assert unique_labels == [0, 1, 2], f"Missing fatigue classes: {unique_labels}"
    print(f" [PASS] All 3 fatigue classes exist: {unique_labels}")

    # Check 8: All 4 environmental classes exist
    unique_envs = sorted(full_df["environment"].unique().tolist())
    assert set(unique_envs) == {"CLEAN", "MODERATE", "POLLUTED", "SPIKE"}, f"Missing env classes: {unique_envs}"
    print(f" [PASS] All 4 environmental classes exist: {unique_envs}")

    # Check 9: Every fatigue class contains multiple environmental conditions
    crosstab = pd.crosstab(full_df["label_name"], full_df["environment"])
    for lbl in ["ALERT", "DROWSY", "FATIGUED"]:
        for env in ["CLEAN", "MODERATE", "POLLUTED", "SPIKE"]:
            count = crosstab.loc[lbl, env]
            assert count > 0, f"Empty cross-combination: {lbl} + {env} has 0 samples!"
    print(f" [PASS] Every fatigue class co-occurs across all 4 environmental conditions.")

    # Check 10: Multiple subjects and sessions exist
    num_subj = full_df["subject_id"].nunique()
    num_sess = full_df["session_id"].nunique()
    assert num_subj == NUM_SUBJECTS and num_sess == TOTAL_SESSIONS
    print(f" [PASS] Structure confirmed: {num_subj} subjects and {num_sess} sessions.")

    # Save to CSV
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated and saved dataset to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(full_df)}")
    print(f"Number of subjects: {num_subj}")
    print(f"Number of sessions: {num_sess}")
    print("\nClass distribution:")
    for lbl_name, count in full_df["label_name"].value_counts().items():
        pct = (count / len(full_df)) * 100
        print(f"  {lbl_name:8s}: {count:5d} ({pct:5.1f}%)")

    print("\nEnvironment distribution:")
    for env_name, count in full_df["environment"].value_counts().items():
        pct = (count / len(full_df)) * 100
        print(f"  {env_name:8s}: {count:5d} ({pct:5.1f}%)")

    print("\nFatigue Label vs. Environment Cross-tabulation:")
    print(crosstab.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
