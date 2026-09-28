#!/usr/bin/env python3
"""
rpi_helmet_edge.py
------------------
Real-Time Edge Sensor Acquisition and Fatigue Inference Loop for Raspberry Pi.

Hardware Setup:
1. InvenSense MPU6050 (I2C address: 0x68):
   - VCC -> 3.3V (Pin 1)
   - GND -> Ground (Pin 6)
   - SDA -> GPIO 2 (SDA, Pin 3)
   - SCL -> GPIO 3 (SCL, Pin 5)

2. Sensirion SPS30 (I2C address: 0x69):
   - VCC -> 5V (Pin 2 or 4)
   - GND -> Ground (Pin 9)
   - SDA -> GPIO 2 (SDA, Pin 3)
   - SCL -> GPIO 3 (SCL, Pin 5)

3. Optional Haptic Buzzer / Alarm LED:
   - Positive/Signal -> GPIO 18 (Pin 12)
   - Negative -> Ground (Pin 14)

Usage:
  # Normal real-time mode with real hardware sensors:
  python3 rpi_helmet_edge.py

  # Bench testing mode (simulates sensor inputs if hardware is not yet wired):
  python3 rpi_helmet_edge.py --mock-hardware

  # Custom alert threshold:
  python3 rpi_helmet_edge.py --alert-threshold 0.70 --alert-pin 18
"""

import os
import sys
import time
import argparse
import numpy as np

# Import core inference logic
from predict import predict_fatigue, load_model_and_features

# Check optional Raspberry Pi hardware libraries
try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# =====================================================================
# MPU6050 Hardware Driver (I2C)
# =====================================================================
class MPU6050Driver:
    """I2C driver for InvenSense MPU6050 6-DoF IMU."""
    MPU_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    # Accelerometer scale factor for +/- 2g: 16384 LSB/g
    ACCEL_SCALE = 16384.0
    # Gyroscope scale factor for +/- 250 deg/s: 131.0 LSB/(deg/s)
    GYRO_SCALE = 131.0
    GRAVITY = 9.80665

    def __init__(self, bus_num=1):
        self.bus_num = bus_num
        self.bus = None
        self.is_connected = False

    def connect(self):
        if not SMBUS_AVAILABLE:
            raise RuntimeError("smbus2 library is not installed. Install with: pip install smbus2")
        try:
            self.bus = SMBus(self.bus_num)
            # Wake up MPU6050 (clears sleep bit in PWR_MGMT_1)
            self.bus.write_byte_data(self.MPU_ADDR, self.PWR_MGMT_1, 0x00)
            time.sleep(0.05)
            self.is_connected = True
            print(f"[HW] MPU6050 connected successfully on I2C bus {self.bus_num} (0x{self.MPU_ADDR:02X}).")
        except Exception as e:
            self.is_connected = False
            raise ConnectionError(f"Failed to connect to MPU6050 at 0x{self.MPU_ADDR:02X}: {e}")

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(self.MPU_ADDR, reg)
        low = self.bus.read_byte_data(self.MPU_ADDR, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def read_motion(self):
        """Reads accelerometer (m/s^2) and gyroscope (deg/s)."""
        raw_ax = self._read_word_2c(self.ACCEL_XOUT_H)
        raw_ay = self._read_word_2c(self.ACCEL_XOUT_H + 2)
        raw_az = self._read_word_2c(self.ACCEL_XOUT_H + 4)

        raw_gx = self._read_word_2c(self.GYRO_XOUT_H)
        raw_gy = self._read_word_2c(self.GYRO_XOUT_H + 2)
        raw_gz = self._read_word_2c(self.GYRO_XOUT_H + 4)

        # Convert to SI units
        ax = (raw_ax / self.ACCEL_SCALE) * self.GRAVITY
        ay = (raw_ay / self.ACCEL_SCALE) * self.GRAVITY
        az = (raw_az / self.ACCEL_SCALE) * self.GRAVITY

        gx = raw_gx / self.GYRO_SCALE
        gy = raw_gy / self.GYRO_SCALE
        gz = raw_gz / self.GYRO_SCALE

        return {
            "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
            "gx": round(gx, 3), "gy": round(gy, 3), "gz": round(gz, 3)
        }


# =====================================================================
# SPS30 Particulate Sensor Driver
# =====================================================================
class SPS30Driver:
    """Interface for Sensirion SPS30 Optical Particle Sensor."""
    SPS_ADDR = 0x69

    def __init__(self, bus_num=1):
        self.bus_num = bus_num
        self.is_connected = False
        self.device = None

    def connect(self):
        try:
            from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection
            from sensirion_i2c_sps.sps30 import Sps30I2cDevice
            
            transceiver = LinuxI2cTransceiver(f"/dev/i2c-{self.bus_num}")
            self.device = Sps30I2cDevice(I2cConnection(transceiver))
            # Start measurement (mass concentration in float)
            self.device.start_measurement()
            time.sleep(1.0)
            self.is_connected = True
            print(f"[HW] SPS30 connected successfully on I2C bus {self.bus_num} (0x{self.SPS_ADDR:02X}).")
        except Exception as e:
            self.is_connected = False
            # Check fallback to UART or diagnostic message
            print(f"[HW Notice] SPS30 I2C connection not detected ({e}). Using mock/fallback particulate stream.")

    def read_pm(self):
        """Reads PM1.0, PM2.5, PM4.0, PM10 in ug/m^3."""
        if self.is_connected and self.device is not None:
            try:
                values = self.device.read_measured_values()
                # SPS30 returns mass concentrations [mc_1p0, mc_2p5, mc_4p0, mc_10p0, ...]
                pm1 = float(values.mass_concentration_pm1p0.value)
                pm25 = float(values.mass_concentration_pm2p5.value)
                pm4 = float(values.mass_concentration_pm4p0.value)
                pm10 = float(values.mass_concentration_pm10p0.value)
                return {
                    "pm1": round(pm1, 2), "pm25": round(pm25, 2),
                    "pm4": round(pm4, 2), "pm10": round(pm10, 2)
                }
            except Exception as e:
                print(f"[HW Error] Failed to read SPS30 values: {e}")

        # Fallback realistic ambient particulate profile if disconnected
        base = np.random.uniform(22.0, 35.0)
        return {
            "pm1": round(base * 0.65, 2),
            "pm25": round(base, 2),
            "pm4": round(base * 1.25, 2),
            "pm10": round(base * 1.55, 2)
        }


# =====================================================================
# GPIO Haptic / Buzzer Controller
# =====================================================================
class AlertController:
    """Manages physical buzzer/LED triggers on Raspberry Pi GPIO."""
    def __init__(self, pin=18):
        self.pin = pin
        self.active = False
        if GPIO_AVAILABLE:
            try:
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT)
                GPIO.output(self.pin, GPIO.LOW)
                self.active = True
                print(f"[HW] Haptic Buzzer/Alert initialized on GPIO {self.pin}.")
            except Exception as e:
                print(f"[HW Notice] Could not initialize GPIO {pin}: {e}")

    def trigger(self, duration=0.2):
        if self.active:
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.pin, GPIO.LOW)
        else:
            # Terminal beep fallback
            sys.stdout.write('\a')
            sys.stdout.flush()

    def cleanup(self):
        if self.active:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()


# =====================================================================
# Main Edge Loop
# =====================================================================
def run_edge_loop(args):
    print("=" * 65)
    print("SMART HELMET AI: RASPBERRY PI EDGE INFERENCE ENGINE")
    print("=" * 65)

    # 1. Load trained ML model and feature configuration
    print("[INIT] Loading Random Forest model and feature schema...")
    model, feature_names = load_model_and_features()
    print(f"[INIT] Model loaded successfully. Input features: {len(feature_names)}")

    # 2. Initialize hardware
    mpu = MPU6050Driver(bus_num=args.i2c_bus)
    sps = SPS30Driver(bus_num=args.i2c_bus)
    alerter = AlertController(pin=args.alert_pin)

    if not args.mock_hardware:
        try:
            mpu.connect()
        except Exception as e:
            print(f"[WARNING] {e}")
            print("Switching to mock hardware mode for testing. Pass real sensors to disable.")
            args.mock_hardware = True

        if not args.mock_hardware:
            sps.connect()
    else:
        print("[MOCK] Running in bench-test mock mode (simulating sensor values).")

    print("\n--- Starting Real-Time Acquisition Loop (Sampling Rate: 1 Hz) ---")
    print("Press Ctrl+C to stop.\n")

    # State tracking for delta change features
    prev_pm25 = None
    prev_pm10 = None
    sample_count = 0

    try:
        while True:
            loop_start = time.time()
            sample_count += 1

            # A. Read MPU6050
            if not args.mock_hardware and mpu.is_connected:
                motion = mpu.read_motion()
            else:
                # Simulated realistic rider movement
                pitch = np.random.normal(5.0, 3.0)
                # Introduce occasional nodding episode for demonstration
                if sample_count % 15 == 0:
                    pitch = 28.0  # Drowsy nod
                rad = np.radians(pitch)
                motion = {
                    "ax": round(-9.81 * np.sin(rad) + np.random.normal(0, 0.1), 3),
                    "ay": round(np.random.normal(0, 0.15), 3),
                    "az": round(9.81 * np.cos(rad) + np.random.normal(0, 0.2), 3),
                    "gx": round(np.random.normal(0, 1.5), 3),
                    "gy": round(np.random.normal(0, 1.2), 3),
                    "gz": round(np.random.normal(0, 1.8), 3),
                }

            # B. Read SPS30
            pm_data = sps.read_pm()

            # C. Calculate delta changes
            if prev_pm25 is None:
                pm25_change = 0.0
                pm10_change = 0.0
            else:
                pm25_change = round(pm_data["pm25"] - prev_pm25, 2)
                pm10_change = round(pm_data["pm10"] - prev_pm10, 2)

            prev_pm25 = pm_data["pm25"]
            prev_pm10 = pm_data["pm10"]

            # D. Assemble complete sensor reading dictionary
            sensor_sample = {
                **motion,
                **pm_data,
                "pm25_change": pm25_change,
                "pm10_change": pm10_change
            }

            # E. Run Edge Inference (predict_fatigue auto-computes tilt_angle and magnitudes)
            result = predict_fatigue(sensor_sample, model, feature_names)

            pred_state = result["predicted_state"]
            conf = result["confidence"][pred_state]
            tilt = result["features_used"]["tilt_angle"]
            pm25_val = result["features_used"]["pm25"]

            # F. Terminal Status Display
            timestamp_str = time.strftime("%H:%M:%S")
            status_symbol = {
                "ALERT": "🟢 [ALERT]   ",
                "DROWSY": "🟡 [DROWSY]  ",
                "FATIGUED": "🔴 [FATIGUED]"
            }.get(pred_state, "[UNKNOWN] ")

            print(f"[{timestamp_str}] #{sample_count:04d} | {status_symbol} "
                  f"Conf: {conf:4.2f} | Tilt: {tilt:5.1f}° | "
                  f"Acc: ({sensor_sample['ax']:5.2f}, {sensor_sample['ay']:5.2f}, {sensor_sample['az']:5.2f}) | "
                  f"PM2.5: {pm25_val:5.1f} µg/m³")

            # G. Safety Haptic Alarm Trigger
            if pred_state in ("DROWSY", "FATIGUED") and conf >= args.alert_threshold:
                print(f"       >>> ⚠️  SAFETY ALERT: {pred_state} detected with high confidence ({conf:.2f})! Triggering buzzer.")
                alerter.trigger(duration=0.3)

            # Sleep to maintain 1 Hz sample rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 1.0 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[STOP] Stopping Smart Helmet Edge loop.")
    finally:
        alerter.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Smart Helmet Edge Inference")
    parser.add_argument("--i2c-bus", type=int, default=1, help="Raspberry Pi I2C bus number (default: 1)")
    parser.add_argument("--alert-pin", type=int, default=18, help="BCM GPIO pin for buzzer/haptic alert (default: 18)")
    parser.add_argument("--alert-threshold", type=float, default=0.65, help="Confidence threshold to trigger alarm (default: 0.65)")
    parser.add_argument("--mock-hardware", action="store_true", help="Force mock hardware mode for bench testing")
    args = parser.parse_args()

    run_edge_loop(args)


if __name__ == "__main__":
    main()
