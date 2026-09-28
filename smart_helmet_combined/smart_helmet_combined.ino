/*
 ================================================================
              SMART HELMET - DUAL-LAYER FATIGUE PIPELINE
                + REAL-TIME GPS & TURN-BY-TURN HUD
 ================================================================

 SYSTEMS
 -------
 MPU6050  -> 6-DoF Motion (ax, ay, az, gx, gy, gz, tilt angle)
 SPS30    -> Environmental Particulates (PM1, PM2.5, PM4, PM10)
 HC-SR04  -> Rear obstacle radar detection
 MAX7219  -> 8x8 LED Matrix warning display + Turn-by-Turn Arrows
 NEO-6M   -> GPS navigation & live telemetry streaming
 DFPlayer -> Voice alerts and turn-by-turn navigation
 D7       -> Central buzzer for audio alerts

 ================================================================
 TWO-LAYER DECISION MAKING ARCHITECTURE
 ================================================================
 LAYER 1 (Threshold Layer - Arduino):
 - Physical tilt angle threshold (18 deg = Drowsy, 30 deg = Fatigued).
 - Time-held persistence (1.5s = Drowsy, 3.0s = Fatigued).
 - Evaluates instantaneous biomechanical posture.
 - Acts as instant fail-safe layer even if AI is offline.

 LAYER 2 (AI Model Layer - Raspberry Pi via Serial):
 - Multimodal Random Forest classifier running on Raspberry Pi.
 - Evaluates subtle motion patterns, gyro angular rates, and environmental PM
 exposure.
 - Sends live prediction back to Arduino: AI:state,confidence.

 FINAL DECISION FUSION:
 - Fuses Layer 1 (Threshold) and Layer 2 (AI) into finalDecisionState:
   0 = ALERT
   1 = DROWSY
   2 = FATIGUED
 - Controls MAX7219 visual alert, buzzer alarm cadence, and voice prompts.
 - When ALERT, displays GPS Turn-by-Turn navigation arrows (↑, ←, →, U,
 Reached).
 ================================================================
*/

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <AltSoftSerial.h>
#include <DFRobotDFPlayerMini.h>
#include <LedControl.h>
#include <SensirionI2cSps30.h>
#include <SoftwareSerial.h>
#include <TinyGPSPlus.h>
#include <Wire.h>
#include <avr/pgmspace.h>

// ================================================================
// PIN CONFIGURATION (Conserves SRAM with const byte)
// ================================================================
const byte BUZZER_PIN = 7;
const byte TRIG_PIN = 4;
const byte ECHO_PIN = 5;

// MAX7219: DIN -> D2, CLK -> D6, CS -> D3
LedControl matrix(2, 6, 3, 1);

// GPS: AltSoftSerial uses RX=D8, TX=D9 on Uno
AltSoftSerial gpsSerial;
TinyGPSPlus gps;

// DFPLAYER: SoftwareSerial RX=D10, TX=D11
SoftwareSerial dfSerial(10, 11);
DFRobotDFPlayerMini player;
bool dfOK = false;

// SPS30 & MPU6050 (I2C: SDA=A4, SCL=A5)
SensirionI2cSps30 sps30;
bool spsOK = false;

Adafruit_MPU6050 mpu;
bool mpuOK = false;

// ================================================================
// FATIGUE STATE DEFINITIONS (0=ALERT, 1=DROWSY, 2=FATIGUED)
// ================================================================
enum FatigueState : byte {
  STATE_ALERT = 0,
  STATE_DROWSY = 1,
  STATE_FATIGUED = 2
};

// Helper: prints state names from Flash without consuming SRAM
void printStateName(byte s) {
  switch (s) {
  case STATE_ALERT:
    Serial.print(F("ALERT"));
    break;
  case STATE_DROWSY:
    Serial.print(F("DROWSY"));
    break;
  case STATE_FATIGUED:
    Serial.print(F("FATIGUED"));
    break;
  default:
    Serial.print(F("UNKNOWN"));
    break;
  }
}

// ================================================================
// LAYER 1: THRESHOLD SETTINGS & VARIABLES
// ================================================================
const float TILT_DROWSY_ANGLE = 18.0f;   // Min tilt (deg) for drowsy
const float TILT_FATIGUE_ANGLE = 30.0f;  // Min tilt (deg) for severe fatigue
const float TILT_RECOVERY_ANGLE = 14.0f; // Recovery angle to reset to ALERT

const unsigned int DROWSY_HOLD_TIME = 1500;  // 1.5 seconds hold for drowsy
const unsigned int FATIGUE_HOLD_TIME = 3000; // 3.0 seconds hold for fatigue

// MPU Reference Vector (Calibration at startup)
float refAX = 0.0f, refAY = 0.0f, refAZ = 9.81f;
float refMagnitude = 9.81f;
bool mpuCalibrated = false;
unsigned long mpuCalibrationStart = 0;
const unsigned int MPU_CALIBRATION_TIME = 4000;

float currentTiltAngle = 0.0f;
float currentGyroMag = 0.0f;
unsigned long tiltStartTime = 0;
bool headTilting = false;

// Layer 1 Output
byte thresholdState = STATE_ALERT;

// ================================================================
// LAYER 2: AI MODEL INPUTS (FROM RASPBERRY PI)
// ================================================================
byte aiState = STATE_ALERT;
float aiConfidence = 0.0f;
bool aiConnected = false;
unsigned long lastAIRecvTime = 0;
const unsigned int AI_TIMEOUT =
    3500; // Fallback to Layer 1 if no AI packet in 3.5s

// ================================================================
// FINAL FUSED DECISION (LAYER 1 + LAYER 2)
// ================================================================
byte finalDecisionState = STATE_ALERT;
byte prevFinalDecisionState = STATE_ALERT;
unsigned long lastFatigueVoice = 0;
const unsigned int FATIGUE_VOICE_INTERVAL =
    10000; // Min 10s between voice alerts

// ================================================================
// SPS30 ENVIRONMENTAL PARTICULATES
// ================================================================
const uint16_t PM1_LIMIT = 70;
const uint16_t PM25_LIMIT = 80;
const uint16_t PM4_LIMIT = 90;
const uint16_t PM10_LIMIT = 111;

uint16_t pm1 = 0, pm25 = 0, pm4 = 0, pm10 = 0;

enum AirLevel : byte { AIR_NORMAL, AIR_PM1, AIR_PM25, AIR_PM4, AIR_PM10 };
byte airLevel = AIR_NORMAL;

// ================================================================
// HC-SR04 RADAR
// ================================================================
float rearDistance = -1.0f;
unsigned long lastRadarRead = 0;
const byte RADAR_INTERVAL = 150;

const float RADAR_CRITICAL = 30.0f;
const float RADAR_DANGER = 50.0f;
const float RADAR_WARNING = 100.0f;
const float RADAR_CAUTION = 200.0f;
byte radarZone = 0;

// ================================================================
// BUZZER & MATRIX TIMERS
// ================================================================
unsigned long buzzerTimer = 0;
bool buzzerState = false;

byte radarBeepCount = 0;
bool radarBeepActive = false;
unsigned long radarSequenceStart = 0;
unsigned long radarLastToggle = 0;
unsigned long matrixTimer = 0;
bool matrixState = false;

// ================================================================
// GPS NAVIGATION & REAL-TIME TELEMETRY
// ================================================================
float destinationLat = 0.0f;
float destinationLon = 0.0f;
bool destinationSet = false;
unsigned long lastNavigation = 0;
unsigned long lastVoice = 0;
const unsigned int NAV_INTERVAL = 2000;
const unsigned int VOICE_INTERVAL = 15000;

// Turn guidance: 0=None, 1=Straight, 2=Right, 3=Left, 4=Reached, 5=UTurn
byte navTurnDirection = 0;
float navDistanceMeters = 0.0f;

// Serial input buffer (40 bytes is plenty for "DEST:lat,lon" or
// "AI:state,conf")
char inputBuffer[40];

// ================================================================
// MAX7219 PATTERNS (ALL STORED IN PROGMEM TO PRESERVE RAM)
// ================================================================
// Fatigue: Flashing X
const byte PROGMEM PATTERN_FATIGUE[8] = {B10000001, B01000010, B00100100,
                                         B00011000, B00011000, B00100100,
                                         B01000010, B10000001};

// Drowsy: Half-drooping Eye Alert
const byte PROGMEM PATTERN_DROWSY[8] = {B00000000, B00111100, B01000010,
                                        B11111111, B10011001, B01000010,
                                        B00111100, B00000000};

// Radar Patterns
const byte PROGMEM PATTERN_RADAR_WARNING[8] = {B00011000, B00111100, B01111110,
                                               B11011011, B00011000, B00100100,
                                               B01000010, B10000001};

const byte PROGMEM PATTERN_RADAR_DANGER[8] = {B00111100, B01111110, B11111111,
                                              B11011011, B11011011, B11111111,
                                              B01111110, B00111100};

const byte PROGMEM PATTERN_RADAR_CRITICAL[8] = {B10101010, B01010101, B10101010,
                                                B01010101, B10101010, B01010101,
                                                B10101010, B01010101};

// Pollution Patterns
const byte PROGMEM PATTERN_PM1[8] = {B00000000, B00000000, B00011000,
                                     B00111100, B00111100, B00011000,
                                     B00000000, B00000000};

const byte PROGMEM PATTERN_PM25[8] = {B00011000, B00111100, B01111110,
                                      B11011011, B00011000, B00011000,
                                      B00000000, B00011000};

const byte PROGMEM PATTERN_PM4[8] = {B00011000, B00111100, B01111110,
                                     B11111111, B11111111, B01111110,
                                     B00111100, B00011000};

const byte PROGMEM PATTERN_PM10[8] = {B11111111, B10000001, B10111101,
                                      B10100101, B10100101, B10111101,
                                      B10000001, B11111111};

// Turn-by-Turn Navigation Arrow Patterns
const byte PROGMEM PATTERN_ARROW_UP[8] = {B00011000, B00111100, B01111110,
                                          B11011011, B00011000, B00011000,
                                          B00011000, B00011000};

const byte PROGMEM PATTERN_ARROW_LEFT[8] = {B00010000, B00110000, B01111111,
                                            B11111111, B01111111, B00110000,
                                            B00010000, B00000000};

const byte PROGMEM PATTERN_ARROW_RIGHT[8] = {B00001000, B00001100, B11111110,
                                             B11111111, B11111110, B00001100,
                                             B00001000, B00000000};

const byte PROGMEM PATTERN_ARROW_UTURN[8] = {B00111100, B01100110, B01100110,
                                             B01100110, B01101110, B01111100,
                                             B00111000, B00010000};

const byte PROGMEM PATTERN_FLAG[8] = {B00000001, B00000011, B00000111,
                                      B00001111, B00000111, B00000011,
                                      B00000001, B00000001};

void drawPattern(const byte *pattern) {
  for (byte row = 0; row < 8; row++) {
    matrix.setRow(0, row, pgm_read_byte(&pattern[row]));
  }
}

void clearMatrix() { matrix.clearDisplay(0); }

// ================================================================
// MPU6050 MATH & CALIBRATION
// ================================================================
float calculateTiltAngle(float ax, float ay, float az) {
  float currentMagnitude = sqrt(ax * ax + ay * ay + az * az);
  if (currentMagnitude < 0.1f || refMagnitude < 0.1f)
    return 0.0f;

  float dot = ax * refAX + ay * refAY + az * refAZ;
  float cosine = dot / (currentMagnitude * refMagnitude);
  if (cosine > 1.0f)
    cosine = 1.0f;
  if (cosine < -1.0f)
    cosine = -1.0f;

  return acos(cosine) * 180.0f / PI;
}

void calibrateMPU() {
  if (!mpuOK || mpuCalibrated)
    return;

  if (mpuCalibrationStart == 0) {
    mpuCalibrationStart = millis();
    Serial.println(F("[MPU] Calibrating 4s: keep upright..."));
  }

  static float sumAX = 0, sumAY = 0, sumAZ = 0;
  static unsigned int samples = 0;

  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  sumAX += accel.acceleration.x;
  sumAY += accel.acceleration.y;
  sumAZ += accel.acceleration.z;
  samples++;

  if (millis() - mpuCalibrationStart >= MPU_CALIBRATION_TIME) {
    refAX = sumAX / samples;
    refAY = sumAY / samples;
    refAZ = sumAZ / samples;
    refMagnitude = sqrt(refAX * refAX + refAY * refAY + refAZ * refAZ);
    mpuCalibrated = true;
    Serial.println(F("[MPU] Calibration complete."));
  }
}

// ================================================================
// LAYER 1: THRESHOLD-BASED FATIGUE ESTIMATION
// ================================================================
void updateFatigueThresholds(float ax, float ay, float az, float gx, float gy,
                             float gz) {
  currentTiltAngle = calculateTiltAngle(ax, ay, az);
  currentGyroMag = sqrt(gx * gx + gy * gy + gz * gz);

  // Severe tilt zone (>= 30 deg)
  if (currentTiltAngle >= TILT_FATIGUE_ANGLE) {
    if (!headTilting) {
      headTilting = true;
      tiltStartTime = millis();
    }
    unsigned long elapsed = millis() - tiltStartTime;
    if (elapsed >= FATIGUE_HOLD_TIME) {
      thresholdState = STATE_FATIGUED;
    } else if (elapsed >= DROWSY_HOLD_TIME) {
      thresholdState = STATE_DROWSY;
    }
  }
  // Moderate tilt / nodding zone (>= 18 deg)
  else if (currentTiltAngle >= TILT_DROWSY_ANGLE) {
    if (!headTilting) {
      headTilting = true;
      tiltStartTime = millis();
    }
    unsigned long elapsed = millis() - tiltStartTime;
    if (elapsed >= DROWSY_HOLD_TIME) {
      thresholdState = STATE_DROWSY;
    }
  }
  // Upright recovery zone (< 14 deg)
  else if (currentTiltAngle < TILT_RECOVERY_ANGLE) {
    headTilting = false;
    thresholdState = STATE_ALERT;
  }
}

// ================================================================
// SERIAL COMMUNICATION & LAYER 2 (AI RECEPTION + GPS STREAM)
// ================================================================
// Fast single-byte comma printer (saves flash over Serial.print(','))
inline void printComma() { Serial.write(','); }

void sendTelemetryToPi(float ax, float ay, float az, float gx, float gy,
                       float gz) {
  static unsigned long lastTelemTime = 0;
  if (millis() - lastTelemTime < 1000)
    return; // 1 Hz stream to Raspberry Pi
  lastTelemTime = millis();

  // Packet format:
  // TELEM:ax,ay,az,gx,gy,gz,tilt,pm1,pm25,pm4,pm10,threshState,gpsFix,lat,lon,speed,sats,navDist,navDir
  Serial.print(F("TELEM:"));
  Serial.print(ax, 2);
  printComma();
  Serial.print(ay, 2);
  printComma();
  Serial.print(az, 2);
  printComma();
  Serial.print(gx, 2);
  printComma();
  Serial.print(gy, 2);
  printComma();
  Serial.print(gz, 2);
  printComma();
  Serial.print(currentTiltAngle, 1);
  printComma();
  Serial.print(pm1);
  printComma();
  Serial.print(pm25);
  printComma();
  Serial.print(pm4);
  printComma();
  Serial.print(pm10);
  printComma();
  Serial.print(thresholdState);
  printComma();

  // GPS Extensions for Dashboard Map & Tracking
  bool hasFix = gps.location.isValid() && (gps.location.age() < 4000);
  Serial.print(hasFix ? 1 : 0);
  printComma();
  Serial.print(hasFix ? gps.location.lat() : 0.0f, 6);
  printComma();
  Serial.print(hasFix ? gps.location.lng() : 0.0f, 6);
  printComma();
  Serial.print(gps.speed.isValid() ? gps.speed.kmph() : 0.0f, 1);
  printComma();
  Serial.print(gps.satellites.isValid() ? gps.satellites.value() : 0);
  printComma();
  Serial.print((int)navDistanceMeters);
  printComma();
  Serial.println(navTurnDirection);
}

void parseAIPacket(char *payload) {
  // Format: "AI:state,confidence" e.g., "AI:1,0.85" or "AI:DROWSY,0.85"
  int parsedState = -1;
  float parsedConf = 0.0f;

  char *comma = strchr(payload, ',');
  if (comma) {
    *comma = '\0';
    parsedConf = atof(comma + 1);
  }

  if (strcmp(payload, "0") == 0 || strcasecmp(payload, "ALERT") == 0) {
    parsedState = STATE_ALERT;
  } else if (strcmp(payload, "1") == 0 || strcasecmp(payload, "DROWSY") == 0) {
    parsedState = STATE_DROWSY;
  } else if (strcmp(payload, "2") == 0 ||
             strcasecmp(payload, "FATIGUED") == 0) {
    parsedState = STATE_FATIGUED;
  }

  if (parsedState >= 0 && parsedState <= 2) {
    aiState = (byte)parsedState;
    aiConfidence = parsedConf;
    aiConnected = true;
    lastAIRecvTime = millis();
  }
}

void processSerialInput() {
  if (!Serial.available())
    return;

  size_t len =
      Serial.readBytesUntil('\n', inputBuffer, sizeof(inputBuffer) - 1);
  inputBuffer[len] = '\0';
  if (len > 0 && inputBuffer[len - 1] == '\r') {
    inputBuffer[len - 1] = '\0';
  }

  // 1. Layer 2 AI Packet from Raspberry Pi ("AI:...")
  if (strncmp(inputBuffer, "AI:", 3) == 0) {
    parseAIPacket(inputBuffer + 3);
    return;
  }

  // 2. GPS Destination Setting ("lat,lon" or "DEST:lat,lon")
  char *destStr = inputBuffer;
  if (strncmp(inputBuffer, "DEST:", 5) == 0) {
    destStr = inputBuffer + 5;
  }
  char *comma = strchr(destStr, ',');
  if (comma) {
    *comma = '\0';
    float lat = atof(destStr);
    float lon = atof(comma + 1);
    if (lat >= -90.0f && lat <= 90.0f && lon >= -180.0f && lon <= 180.0f) {
      destinationLat = lat;
      destinationLon = lon;
      destinationSet = true;
      lastVoice = 0;
      navTurnDirection = 1; // Start with straight
      Serial.println(F("[NAV] Destination set."));
      return;
    }
  }

  // 3. Status query
  if (strcasecmp(inputBuffer, "STATUS") == 0) {
    Serial.print(F("[STATUS] Thresh="));
    printStateName(thresholdState);
    Serial.print(F(" | AI="));
    if (aiConnected) {
      printStateName(aiState);
    } else {
      Serial.print(F("OFFLINE"));
    }
    Serial.print(F(" | FINAL="));
    printStateName(finalDecisionState);
    Serial.println();
  }
}

// ================================================================
// FINAL DECISION FUSION LAYER (LAYER 1 + LAYER 2)
// ================================================================
void updateFinalDecision() {
  // AI communication watchdog
  if (aiConnected && (millis() - lastAIRecvTime > AI_TIMEOUT)) {
    aiConnected = false;
    Serial.println(F("[SYSTEM] AI timeout -> Layer 1 fallback."));
  }

  if (!aiConnected) {
    finalDecisionState = thresholdState;
  } else {
    // 1. Critical Fatigue Override
    if (thresholdState == STATE_FATIGUED ||
        (aiState == STATE_FATIGUED && aiConfidence >= 0.65f)) {
      finalDecisionState = STATE_FATIGUED;
    }
    // 2. Drowsiness Detection
    else if (thresholdState == STATE_DROWSY ||
             (aiState == STATE_DROWSY && aiConfidence >= 0.55f) ||
             aiState == STATE_FATIGUED) {
      finalDecisionState = STATE_DROWSY;
    }
    // 3. Normal / Attentive
    else {
      finalDecisionState = STATE_ALERT;
    }
  }

  // Voice announcement on state escalation
  if (finalDecisionState != prevFinalDecisionState) {
    if (finalDecisionState == STATE_FATIGUED && dfOK &&
        (millis() - lastFatigueVoice >= FATIGUE_VOICE_INTERVAL)) {
      player.playMp3Folder(7); // Track 7: Critical Fatigue Alarm
      lastFatigueVoice = millis();
    } else if (finalDecisionState == STATE_DROWSY && dfOK &&
               (millis() - lastFatigueVoice >= FATIGUE_VOICE_INTERVAL)) {
      player.playMp3Folder(6); // Track 6: Drowsiness Alert
      lastFatigueVoice = millis();
    }
    prevFinalDecisionState = finalDecisionState;
  }
}

// ================================================================
// MPU6050 PIPELINE MASTER UPDATE
// ================================================================
void updateFatiguePipeline() {
  float ax = 0.0f, ay = 0.0f, az = 9.81f;
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;

  if (mpuOK) {
    if (!mpuCalibrated) {
      calibrateMPU();
    } else {
      sensors_event_t accel, gyro, temp;
      mpu.getEvent(&accel, &gyro, &temp);

      ax = accel.acceleration.x;
      ay = accel.acceleration.y;
      az = accel.acceleration.z;
      gx = gyro.gyro.x * 180.0f / PI; // convert rad/s to deg/s
      gy = gyro.gyro.y * 180.0f / PI;
      gz = gyro.gyro.z * 180.0f / PI;

      // 1. Compute Layer 1 Threshold Decision
      updateFatigueThresholds(ax, ay, az, gx, gy, gz);
    }
  } else {
    finalDecisionState = STATE_ALERT;
  }

  // 2. Stream telemetry (Kinematics + SPS30 + GPS) to Raspberry Pi AI
  sendTelemetryToPi(ax, ay, az, gx, gy, gz);

  // 3. Compute Final Fused Decision
  updateFinalDecision();
}

// ================================================================
// HC-SR04 RADAR
// ================================================================
float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  if (duration == 0)
    return -1.0f;

  float distance = duration * 0.0343f / 2.0f;
  if (distance < 2.0f || distance > 400.0f)
    return -1.0f;
  return distance;
}

byte determineRadarZone(float distance) {
  if (distance < 0.0f)
    return 0;
  if (distance <= RADAR_CRITICAL)
    return 4;
  if (distance <= RADAR_DANGER)
    return 3;
  if (distance <= RADAR_WARNING)
    return 2;
  if (distance <= RADAR_CAUTION)
    return 1;
  return 0;
}

void updateRadar() {
  if (millis() - lastRadarRead < RADAR_INTERVAL)
    return;
  lastRadarRead = millis();

  float newDistance = readDistance();
  if (newDistance < 0.0f) {
    rearDistance = -1.0f;
    radarZone = 0;
    return;
  }

  rearDistance = newDistance;
  radarZone = determineRadarZone(rearDistance);
}

// ================================================================
// SPS30 PARTICULATE SENSOR
// ================================================================
void updateSPS30() {
  if (!spsOK)
    return;

  uint16_t error;
  uint16_t ready = 0;
  error = sps30.readDataReadyFlag(ready);
  if (error != 0 || !ready)
    return;

  uint16_t nc0, nc1, nc2, nc4, nc10, typical;
  error = sps30.readMeasurementValuesUint16(pm1, pm25, pm4, pm10, nc0, nc1, nc2,
                                            nc4, nc10, typical);
  if (error != 0)
    return;

  float ratio1 = (float)pm1 / PM1_LIMIT;
  float ratio25 = (float)pm25 / PM25_LIMIT;
  float ratio4 = (float)pm4 / PM4_LIMIT;
  float ratio10 = (float)pm10 / PM10_LIMIT;

  float highest = 1.0f;
  airLevel = AIR_NORMAL;

  if (ratio1 > highest) {
    highest = ratio1;
    airLevel = AIR_PM1;
  }
  if (ratio25 > highest) {
    highest = ratio25;
    airLevel = AIR_PM25;
  }
  if (ratio4 > highest) {
    highest = ratio4;
    airLevel = AIR_PM4;
  }
  if (ratio10 > highest) {
    highest = ratio10;
    airLevel = AIR_PM10;
  }
}

// ================================================================
// MAX7219 MATRIX UPDATE
// Priority: 1. FATIGUE -> 2. DROWSY -> 3. AIR -> 4. RADAR -> 5. NAVIGATION
// ================================================================
void updateMatrix() {
  // 1. FATIGUED ALERT (Highest priority: Flashing X)
  if (finalDecisionState == STATE_FATIGUED) {
    if (millis() - matrixTimer >= 200) {
      matrixTimer = millis();
      matrixState = !matrixState;
    }
    if (matrixState)
      drawPattern(PATTERN_FATIGUE);
    else
      clearMatrix();
    return;
  }

  // 2. DROWSY ALERT
  if (finalDecisionState == STATE_DROWSY) {
    if (millis() - matrixTimer >= 350) {
      matrixTimer = millis();
      matrixState = !matrixState;
    }
    if (matrixState)
      drawPattern(PATTERN_DROWSY);
    else
      clearMatrix();
    return;
  }

  // 3. AIR POLLUTION WARNING
  if (airLevel != AIR_NORMAL) {
    if (millis() - matrixTimer >= 400) {
      matrixTimer = millis();
      matrixState = !matrixState;
    }
    if (!matrixState) {
      clearMatrix();
      return;
    }
    switch (airLevel) {
    case AIR_PM1:
      drawPattern(PATTERN_PM1);
      break;
    case AIR_PM25:
      drawPattern(PATTERN_PM25);
      break;
    case AIR_PM4:
      drawPattern(PATTERN_PM4);
      break;
    case AIR_PM10:
      drawPattern(PATTERN_PM10);
      break;
    }
    return;
  }

  // 4. RADAR PROXIMITY
  if (radarZone > 0) {
    if (millis() - matrixTimer >= 300) {
      matrixTimer = millis();
      matrixState = !matrixState;
    }
    if (!matrixState) {
      clearMatrix();
      return;
    }
    if (radarZone <= 2)
      drawPattern(PATTERN_RADAR_WARNING);
    else if (radarZone == 3)
      drawPattern(PATTERN_RADAR_DANGER);
    else
      drawPattern(PATTERN_RADAR_CRITICAL);
    return;
  }

  // 5. GPS TURN-BY-TURN HUD (When riding normally and destination active)
  if (destinationSet && navTurnDirection > 0) {
    switch (navTurnDirection) {
    case 1:
      drawPattern(PATTERN_ARROW_UP);
      break; // Straight
    case 2:
      drawPattern(PATTERN_ARROW_RIGHT);
      break; // Right Turn
    case 3:
      drawPattern(PATTERN_ARROW_LEFT);
      break; // Left Turn
    case 4:
      drawPattern(PATTERN_FLAG);
      break; // Destination Reached
    case 5:
      drawPattern(PATTERN_ARROW_UTURN);
      break; // U-Turn
    }
    return;
  }

  // Standby: Clear matrix
  clearMatrix();
}

// ================================================================
// BUZZER UPDATE
// ================================================================
void updateBuzzer() {
  // 1. FATIGUE: Continuous rapid alarm (150ms on / 150ms off)
  if (finalDecisionState == STATE_FATIGUED) {
    if (millis() - buzzerTimer >= 150) {
      buzzerTimer = millis();
      buzzerState = !buzzerState;
      digitalWrite(BUZZER_PIN, buzzerState);
    }
    radarBeepCount = 0;
    radarBeepActive = false;
    return;
  }

  // 2. DROWSY: 300ms cadence
  if (finalDecisionState == STATE_DROWSY) {
    if (millis() - buzzerTimer >= 300) {
      buzzerTimer = millis();
      buzzerState = !buzzerState;
      digitalWrite(BUZZER_PIN, buzzerState);
    }
    radarBeepCount = 0;
    radarBeepActive = false;
    return;
  }

  // 3. AIR POLLUTION
  if (airLevel != AIR_NORMAL) {
    unsigned int interval = 1200;
    if (airLevel == AIR_PM10)
      interval = 200;
    else if (airLevel == AIR_PM4)
      interval = 350;
    else if (airLevel == AIR_PM25)
      interval = 650;

    if (millis() - buzzerTimer >= interval) {
      buzzerTimer = millis();
      buzzerState = !buzzerState;
      digitalWrite(BUZZER_PIN, buzzerState);
    }
    return;
  }

  // 4. RADAR
  if (radarZone == 0) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerState = false;
    radarBeepCount = 0;
    radarBeepActive = false;
    return;
  }

  if (radarZone == 4) { // Critical: <= 30cm
    if (millis() - buzzerTimer >= 100) {
      buzzerTimer = millis();
      buzzerState = !buzzerState;
      digitalWrite(BUZZER_PIN, buzzerState);
    }
    return;
  }

  if (radarZone == 3) { // Danger: 30-50cm
    if (millis() - buzzerTimer >= 250) {
      buzzerTimer = millis();
      buzzerState = !buzzerState;
      digitalWrite(BUZZER_PIN, buzzerState);
    }
    return;
  }

  if (radarZone == 2) { // Warning: 50-100cm (2 beeps)
    if (!radarBeepActive) {
      radarBeepActive = true;
      radarBeepCount = 0;
      radarSequenceStart = millis();
      radarLastToggle = millis();
      buzzerState = false;
    }
    if (radarBeepCount < 4) {
      if (millis() - radarLastToggle >= 150) {
        radarLastToggle = millis();
        buzzerState = !buzzerState;
        digitalWrite(BUZZER_PIN, buzzerState);
        radarBeepCount++;
      }
    } else {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerState = false;
      if (millis() - radarSequenceStart >= 5000)
        radarBeepActive = false;
    }
    return;
  }

  if (radarZone == 1) { // Caution: 100-200cm (1 beep)
    if (!radarBeepActive) {
      radarBeepActive = true;
      radarBeepCount = 0;
      radarSequenceStart = millis();
      digitalWrite(BUZZER_PIN, HIGH);
      buzzerState = true;
      radarLastToggle = millis();
    }
    if (buzzerState && millis() - radarLastToggle >= 200) {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerState = false;
    }
    if (millis() - radarSequenceStart >= 5000)
      radarBeepActive = false;
    return;
  }

  digitalWrite(BUZZER_PIN, LOW);
  buzzerState = false;
}

// ================================================================
// GPS UPDATE & TURN-BY-TURN COMPUTATION
// ================================================================
void updateGPS() {
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
  }
}

void updateNavigation() {
  if (!destinationSet || !gps.location.isValid()) {
    navTurnDirection = 0;
    return;
  }
  if (millis() - lastNavigation < NAV_INTERVAL)
    return;
  lastNavigation = millis();

  float currentLat = (float)gps.location.lat();
  float currentLon = (float)gps.location.lng();
  navDistanceMeters = (float)TinyGPSPlus::distanceBetween(
      currentLat, currentLon, destinationLat, destinationLon);
  float destinationBearing = (float)TinyGPSPlus::courseTo(
      currentLat, currentLon, destinationLat, destinationLon);

  // Destination reached within 20 meters
  if (navDistanceMeters <= 20.0f) {
    navTurnDirection = 4; // Reached
    if (dfOK && millis() - lastVoice >= VOICE_INTERVAL) {
      player.playMp3Folder(4); // Track 4: Destination reached
      lastVoice = millis();
    }
    return;
  }

  if (!gps.course.isValid() || gps.speed.kmph() < 2.0f)
    return;

  float currentCourse = (float)gps.course.deg();
  float difference = destinationBearing - currentCourse;
  while (difference > 180.0f)
    difference -= 360.0f;
  while (difference < -180.0f)
    difference += 360.0f;

  if (abs(difference) <= 25.0f) {
    navTurnDirection = 1; // Straight ahead
  } else if (difference > 25.0f && difference < 150.0f) {
    navTurnDirection = 2; // Turn Right
  } else if (difference < -25.0f && difference > -150.0f) {
    navTurnDirection = 3; // Turn Left
  } else {
    navTurnDirection = 5; // U-Turn
  }

  if (dfOK && millis() - lastVoice >= VOICE_INTERVAL) {
    player.playMp3Folder(navTurnDirection);
    lastVoice = millis();
  }
}

// ================================================================
// SETUP
// ================================================================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  matrix.shutdown(0, false);
  matrix.setIntensity(0, 5);
  matrix.clearDisplay(0);

  Wire.begin();
  // Safe timeout: 25ms allows Sensirion SPS30 clock stretching without
  // resetting
  Wire.setWireTimeout(25000, true);
  delay(100);

  // Quick I2C bus diagnostic scan
  Serial.print(F("[I2C SCAN] Found: "));
  byte i2cFound = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print(F("0x"));
      if (addr < 16)
        Serial.print('0');
      Serial.print(addr, HEX);
      Serial.print(' ');
      i2cFound++;
    }
  }
  if (i2cFound == 0)
    Serial.print(F("NONE!"));
  Serial.println();

  // MPU6050 (auto-detect 0x68 or 0x69)
  if (mpu.begin(0x68, &Wire)) {
    mpuOK = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println(F("[HW] MPU6050 OK (0x68)"));
  } else if (mpu.begin(0x69, &Wire)) {
    mpuOK = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println(F("[HW] MPU6050 OK (0x69)"));
  } else {
    Serial.println(F("[HW] MPU6050 FAIL"));
  }

  // SPS30
  sps30.begin(Wire, 0x69);
  sps30.stopMeasurement();
  delay(50);
  if (sps30.startMeasurement(SPS30_OUTPUT_FORMAT_OUTPUT_FORMAT_UINT16) == 0) {
    spsOK = true;
    Serial.println(F("[HW] SPS30 OK"));
  } else {
    Serial.println(F("[HW] SPS30 FAIL"));
  }

  // GPS (NEO-6M on D8 RX, D9 TX)
  gpsSerial.begin(9600);
  Serial.println(F("[HW] GPS READY"));

  // DFPlayer Mini Voice
  dfSerial.begin(9600);
  delay(300);
  if (player.begin(dfSerial)) {
    dfOK = true;
    player.outputDevice(DFPLAYER_DEVICE_SD);
    player.volume(25);
    Serial.println(F("[HW] DFPLAYER OK"));
  } else {
    Serial.println(F("[HW] DFPLAYER FAIL"));
  }

  Serial.println(F("SMART HELMET DUAL-LAYER AI READY"));
}

// ================================================================
// MAIN LOOP
// ================================================================
void loop() {
  updateGPS();
  processSerialInput();
  updateNavigation();

  // Dual-Layer Fatigue Pipeline (Thresholds + USB AI Streaming + Fusion)
  updateFatiguePipeline();

  updateSPS30();
  updateRadar();
  updateMatrix();
  updateBuzzer();
}
