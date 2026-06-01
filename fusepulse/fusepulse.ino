/*
 * ============================================================
 *  FusePulse — Cuffless BP Monitor  (v1.1 — all bugs fixed)
 *  Hardware : ESP32 + 2x MAX30102 + MPU-6050 + SSD1306 OLED
 *  Method   : Dual-site PTT  +  Moens-Korteweg formula
 *  WiFi     : AP mode  SSID: FusePulse-BP  Pass: bp123456
 *  Dashboard: http://192.168.4.1
 *
 *  Libraries (Library Manager):
 *    SparkFun MAX3010x | Adafruit SSD1306 | Adafruit GFX
 *    ArduinoJson v6   | arduinoWebSockets (Markus Sattler)
 *
 *  FIXES vs v1.0:
 *   1. DC baseline cold-start — fast-init on first valid sample
 *   2. Signal state reset when finger removed/replaced
 *   3. requestFrom() cast — prevents I2C silent failure on ESP32 3.x
 *   4. Removed unused streamW/streamF arrays (freed 640 bytes RAM)
 *   5. pushWS() String() heap fragmentation — replaced with sprintf
 *   6. Adaptive peak threshold — works on dark skin / poor placement
 *   7. 3-second warmup guard — no PTT calc until DC has converged
 *   8. Integer overflow in checkMotion ax*ax (int16→int32 cast)
 *   9. OLED page only rotates when finger is actually on
 * ============================================================
 */

#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include "web_interface.h"

// ============================================================
// PIN MAP
// ============================================================
#define I2C0_SDA  21    // Bus0: Wrist MAX30102 + OLED + MPU-6050
#define I2C0_SCL  22
#define I2C1_SDA  25    // Bus1: Finger MAX30102 (same addr 0x57, different bus)
#define I2C1_SCL  26
#define BUZZER    27    // Optional active buzzer

// ============================================================
// OBJECTS
// ============================================================
TwoWire Bus0 = TwoWire(0);
TwoWire Bus1 = TwoWire(1);

MAX30105             wristPPG;
MAX30105             fingerPPG;
Adafruit_SSD1306     oled(128, 64, &Bus0, -1);
WebServer            http(80);
WebSocketsServer     ws(81);

// ============================================================
// WiFi AP
// ============================================================
const char* AP_SSID = "FusePulse-BP";
const char* AP_PASS = "bp123456";

// ============================================================
// SIGNAL PROCESSING CONSTANTS
// ============================================================
#define SAMPLE_US       10000UL   // 100 Hz
#define DC_ALPHA        0.001f    // Slow DC tracker (steady state)
#define DC_ALPHA_FAST   0.10f     // Fast DC tracker (cold-start / re-place)
#define SMOOTH_WIN      5
#define PEAK_REFRACT    400       // ms  (~150 BPM max)
#define MIN_IR          50000UL   // Finger-presence threshold
#define WARMUP_MS       3000      // Discard PTT for 3 s after finger placed
#define ARTERY_CM       20.0f     // Wrist → fingertip arterial distance

// ============================================================
// SIGNAL STATE
// ============================================================
float wDC = 0,  fDC = 0;
bool  dcReady   = false;          // true once fast-init converged
int   dcInitCnt = 0;              // samples counted during fast-init
#define DC_INIT_SAMPLES 30        // 0.3 s at 100 Hz

float wSmoothBuf[SMOOTH_WIN] = {0};
float fSmoothBuf[SMOOTH_WIN] = {0};
int   smoothIdx = 0;

float wPrev = 0, wPPrev = 0;
float fPrev = 0, fPPrev = 0;

// Adaptive peak threshold — tracks recent signal max
float wPeakMax  = 0.001f;
float fPeakMax  = 0.001f;
#define PEAK_THRESH_RATIO  0.3f   // peak must be > 30 % of recent max

// ============================================================
// FINGER STATE — detect place/remove transitions
// ============================================================
bool  fingerOn      = false;
bool  fingerWasOn   = false;
unsigned long fingerOnTime = 0;   // millis() when finger was placed

// ============================================================
// PEAK TRACKING
// ============================================================
unsigned long lastWristPeak  = 0;
unsigned long lastFingerPeak = 0;

#define PTT_HIST  8
float pttBuf[PTT_HIST];
int   pttIdx = 0;

// ============================================================
// VITALS
// ============================================================
float sbp = 0, dbp = 0, hr = 0;
float pttMs = 0, pwv = 0, asi = 0;
bool  motion = false;

// ============================================================
// CALIBRATION
// ============================================================
bool  calibrated = false;
float calSBP = 120.0f, calDBP = 80.0f, calPTT = 0.0f;
#define ALPHA_SBP  0.7f
#define ALPHA_DBP  0.4f

// ============================================================
// OLED + TIMING
// ============================================================
int  oledPage = 0;
unsigned long tLastOled = 0, tLastPage = 0;
unsigned long tLastSample = 0, tLastWS = 0;
bool wsClient = false;

// ============================================================
// MPU
// ============================================================
#define MPU_ADDR 0x68

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== FusePulse BP Monitor v1.1 ===");

  Bus0.begin(I2C0_SDA, I2C0_SCL, 400000);
  Bus1.begin(I2C1_SDA, I2C1_SCL, 400000);

  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, LOW);

  initOLED();
  initPPG();
  initMPU();
  initWiFi();
  initHTTP();
  initWS();

  Serial.printf("WiFi: %s  |  http://192.168.4.1\n", AP_SSID);
  oledMsg("Connect WiFi:", AP_SSID, AP_PASS);
  delay(2500);
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  http.handleClient();
  ws.loop();

  if (micros() - tLastSample >= SAMPLE_US) {
    tLastSample = micros();
    sampleLoop();
  }

  if (millis() - tLastOled > 500) {
    tLastOled = millis();
    drawOLED();
  }

  // FIX 9 — only rotate page while finger is on
  if (fingerOn && millis() - tLastPage > 3000) {
    tLastPage = millis();
    oledPage  = (oledPage + 1) % 3;
  }
}

// ============================================================
// SAMPLE LOOP  (100 Hz)
// ============================================================
void sampleLoop() {
  wristPPG.check();
  fingerPPG.check();

  if (!wristPPG.available() || !fingerPPG.available()) return;

  uint32_t rawW = wristPPG.getIR();
  uint32_t rawF = fingerPPG.getIR();
  wristPPG.nextSample();
  fingerPPG.nextSample();

  fingerOn = (rawW > MIN_IR && rawF > MIN_IR);

  // ── FIX 2 — detect finger placed/removed transitions ──────
  if (fingerOn && !fingerWasOn) {
    // Finger just placed — reset all signal state
    resetSignalState(rawW, rawF);
    fingerOnTime = millis();
    Serial.println("[FP] Finger placed — resetting signal state");
  }
  if (!fingerOn && fingerWasOn) {
    Serial.println("[FP] Finger removed");
    sbp = dbp = hr = pttMs = pwv = asi = 0;
  }
  fingerWasOn = fingerOn;

  if (!fingerOn) return;

  // ── FIX 1 — DC baseline: fast-init then slow tracking ─────
  if (!dcReady) {
    // Fast convergence for first DC_INIT_SAMPLES samples
    wDC = wDC * (1.0f - DC_ALPHA_FAST) + rawW * DC_ALPHA_FAST;
    fDC = fDC * (1.0f - DC_ALPHA_FAST) + rawF * DC_ALPHA_FAST;
    dcInitCnt++;
    if (dcInitCnt >= DC_INIT_SAMPLES) dcReady = true;
    return;  // don't process signal until DC has settled
  }
  // Slow tracking in steady state
  wDC = wDC * (1.0f - DC_ALPHA) + rawW * DC_ALPHA;
  fDC = fDC * (1.0f - DC_ALPHA) + rawF * DC_ALPHA;

  // AC component (normalised, removes DC offset)
  float wAC = (rawW - wDC) / (wDC + 1.0f);
  float fAC = (rawF - fDC) / (fDC + 1.0f);

  // Moving-average smoothing
  wSmoothBuf[smoothIdx] = wAC;
  fSmoothBuf[smoothIdx] = fAC;
  smoothIdx = (smoothIdx + 1) % SMOOTH_WIN;
  float ws = 0, fs = 0;
  for (int i = 0; i < SMOOTH_WIN; i++) { ws += wSmoothBuf[i]; fs += fSmoothBuf[i]; }
  ws /= SMOOTH_WIN;
  fs /= SMOOTH_WIN;

  // FIX 6 — update adaptive peak envelope
  wPeakMax = wPeakMax * 0.999f + fabsf(ws) * 0.001f;
  fPeakMax = fPeakMax * 0.999f + fabsf(fs) * 0.001f;
  if (fabsf(ws) > wPeakMax) wPeakMax = fabsf(ws);
  if (fabsf(fs) > fPeakMax) fPeakMax = fabsf(fs);

  // FIX 7 — skip PTT during warmup (DC not fully converged yet)
  bool warmedUp = (millis() - fingerOnTime > WARMUP_MS);

  if (warmedUp) detectPeaks(ws, fs);

  checkMotion();

  if (wsClient && millis() - tLastWS > 50) {
    tLastWS = millis();
    pushWS(ws * 1000.0f, fs * 1000.0f);
  }
}

// ============================================================
// SIGNAL STATE RESET  (FIX 2)
// ============================================================
void resetSignalState(uint32_t rawW, uint32_t rawF) {
  // Seed DC immediately to current raw value so first AC is near zero
  wDC = (float)rawW;
  fDC = (float)rawF;
  dcReady   = false;
  dcInitCnt = 0;

  // Clear smooth buffers
  for (int i = 0; i < SMOOTH_WIN; i++) {
    wSmoothBuf[i] = 0;
    fSmoothBuf[i] = 0;
  }
  smoothIdx = 0;

  // Clear peak history
  wPrev = wPPrev = 0;
  fPrev = fPPrev = 0;
  wPeakMax = fPeakMax = 0.001f;
  lastWristPeak  = 0;
  lastFingerPeak = 0;
  pttIdx = 0;
  for (int i = 0; i < PTT_HIST; i++) pttBuf[i] = 0;
}

// ============================================================
// PEAK DETECTION  (3-point local-max + adaptive threshold)
// ============================================================
void detectPeaks(float w, float f) {
  unsigned long now = millis();

  // FIX 6 — adaptive thresholds based on recent signal amplitude
  float wThresh = wPeakMax * PEAK_THRESH_RATIO;
  float fThresh = fPeakMax * PEAK_THRESH_RATIO;

  // ── WRIST peak ────────────────────────────────────────────
  if (wPrev > wPPrev && wPrev > w && wPrev > wThresh) {
    if (now - lastWristPeak > PEAK_REFRACT) {
      if (lastWristPeak > 0) {
        float ibi = (float)(now - lastWristPeak);
        if (ibi > 300 && ibi < 2000) hr = 60000.0f / ibi;
      }
      lastWristPeak = now;
    }
  }

  // ── FINGER peak ───────────────────────────────────────────
  if (fPrev > fPPrev && fPrev > f && fPrev > fThresh) {
    if (now - lastFingerPeak > PEAK_REFRACT) {
      lastFingerPeak = now;

      if (lastWristPeak > 0) {
        long ptt = (long)now - (long)lastWristPeak;
        if (ptt >= 50 && ptt <= 500) {
          pttBuf[pttIdx % PTT_HIST] = (float)ptt;
          pttIdx++;
          calcVitals();
        }
      }
    }
  }

  wPPrev = wPrev;  wPrev = w;
  fPPrev = fPrev;  fPrev = f;
}

// ============================================================
// BP CALCULATION
// ============================================================
void calcVitals() {
  int n = (pttIdx < PTT_HIST) ? pttIdx : PTT_HIST;
  if (n == 0) return;

  float sum = 0;
  for (int i = 0; i < n; i++) sum += pttBuf[i];
  float avgPTT = sum / n;
  pttMs = avgPTT;

  float ptt_s = avgPTT / 1000.0f;
  pwv = (ARTERY_CM / 100.0f) / ptt_s;
  asi = pwv;

  if (!calibrated) {
    sbp = (9.0f  / ptt_s) + 88.0f;
    dbp = (5.0f  / ptt_s) + 50.0f;
    sbp = constrain(sbp, 80.0f, 180.0f);
    dbp = constrain(dbp, 50.0f, 120.0f);
  } else {
    float dPTT = calPTT - avgPTT;
    sbp = calSBP + ALPHA_SBP * dPTT;
    dbp = calDBP + ALPHA_DBP * dPTT;
    sbp = constrain(sbp, 80.0f, 200.0f);
    dbp = constrain(dbp, 50.0f, 130.0f);
  }

  Serial.printf("[FP] PTT:%.1fms SBP:%.0f DBP:%.0f HR:%.0f PWV:%.2fm/s\n",
                pttMs, sbp, dbp, hr, pwv);

  if (sbp >= 140.0f || dbp >= 90.0f) tone(BUZZER, 1000, 200);
}

// ============================================================
// MPU-6050
// ============================================================
void initMPU() {
  Bus0.beginTransmission(MPU_ADDR);
  Bus0.write(0x6B);
  Bus0.write(0x00);
  if (Bus0.endTransmission() == 0) Serial.println("MPU-6050 OK");
  else                             Serial.println("MPU-6050 not found (motion disabled)");
}

void checkMotion() {
  static unsigned long t = 0;
  if (millis() - t < 80) return;
  t = millis();

  Bus0.beginTransmission((uint8_t)MPU_ADDR);
  Bus0.write(0x3B);
  Bus0.endTransmission(false);
  // FIX 3 — explicit uint8_t casts prevent wrong overload on ESP32 3.x
  Bus0.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6, (bool)true);

  int16_t ax = ((int16_t)Bus0.read() << 8) | Bus0.read();
  int16_t ay = ((int16_t)Bus0.read() << 8) | Bus0.read();
  int16_t az = ((int16_t)Bus0.read() << 8) | Bus0.read();

  // FIX 8 — cast to int32_t before squaring to prevent overflow
  // int16 max = 32768; 32768² = 1,073,741,824 > INT16_MAX
  float mag = sqrtf((float)((int32_t)ax*(int32_t)ax +
                             (int32_t)ay*(int32_t)ay +
                             (int32_t)az*(int32_t)az)) / 16384.0f;
  motion = (fabsf(mag - 1.0f) > 0.30f);
}

// ============================================================
// PPG INIT
// ============================================================
void initPPG() {
  if (wristPPG.begin(Bus0, I2C_SPEED_FAST)) {
    wristPPG.setup(60, 4, 2, 100, 69, 4096);
    wristPPG.setPulseAmplitudeRed(0);
    wristPPG.setPulseAmplitudeGreen(0);
    Serial.println("Wrist MAX30102 OK");
  } else Serial.println("Wrist MAX30102 FAIL");

  if (fingerPPG.begin(Bus1, I2C_SPEED_FAST)) {
    fingerPPG.setup(60, 4, 2, 100, 69, 4096);
    fingerPPG.setPulseAmplitudeRed(0);
    fingerPPG.setPulseAmplitudeGreen(0);
    Serial.println("Finger MAX30102 OK");
  } else Serial.println("Finger MAX30102 FAIL");
}

// ============================================================
// OLED
// ============================================================
void initOLED() {
  if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found"); return;
  }
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(18, 8);  oled.print("FusePulse  v1.1");
  oled.setCursor(8,  24); oled.print("Cuffless BP Monitor");
  oled.setCursor(20, 44); oled.print("Initialising...");
  oled.display();
}

void oledMsg(const char* l1, const char* l2, const char* l3) {
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setCursor(0, 2);  oled.print(l1);
  oled.setCursor(0, 20); oled.print(l2);
  oled.setCursor(0, 40); oled.print(l3);
  oled.display();
}

void drawOLED() {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);

  if (motion) { oled.setCursor(84, 0); oled.print("! MOTION"); }

  if (!fingerOn) {
    // FIX 9 — reset page to 0 when finger removed so it starts fresh
    oledPage = 0;
    oled.setCursor(14, 24); oled.print("Place finger +");
    oled.setCursor(10, 38); oled.print("wrist on sensors");
    oled.display();
    return;
  }

  // Show "warming up" during DC init + 3s grace period
  if (!dcReady || (millis() - fingerOnTime < WARMUP_MS)) {
    oled.setCursor(0,  0); oled.print("Signal stabilising");
    oled.setCursor(0, 20); oled.print("Hold still...");
    uint32_t elapsed = millis() - fingerOnTime;
    uint32_t pct     = (elapsed * 100) / (WARMUP_MS + 300);
    if (pct > 100) pct = 100;
    oled.drawRect(0, 40, 128, 10, SSD1306_WHITE);
    oled.fillRect(0, 40, (int)(pct * 128 / 100), 10, SSD1306_WHITE);
    oled.display();
    return;
  }

  char buf[24];
  switch (oledPage) {
    case 0:
      oled.setCursor(0, 0);
      oled.print(calibrated ? "BP (calibrated)" : "BP (pre-cal est)");
      oled.setTextSize(2);
      oled.setCursor(0, 16);
      if (sbp > 0) { snprintf(buf, sizeof(buf), "%.0f/%.0f", sbp, dbp); oled.print(buf); }
      else           oled.print("-- / --");
      oled.setTextSize(1);
      oled.setCursor(0, 50); oled.print("mmHg");
      oled.setCursor(72, 50);
      snprintf(buf, sizeof(buf), "HR %.0f bpm", hr);
      oled.print(buf);
      break;

    case 1:
      oled.setCursor(0,  0); oled.print("Pulse Transit Time");
      oled.setCursor(0, 14);
      snprintf(buf, sizeof(buf), "PTT : %.1f ms",  pttMs); oled.print(buf);
      oled.setCursor(0, 28);
      snprintf(buf, sizeof(buf), "PWV : %.2f m/s", pwv);   oled.print(buf);
      oled.setCursor(0, 42); oled.print("(wrist -> finger)");
      break;

    case 2:
      oled.setCursor(0,  0); oled.print("Vascular Status");
      oled.setCursor(0, 14);
      snprintf(buf, sizeof(buf), "ASI : %.2f m/s", asi);   oled.print(buf);
      oled.setCursor(0, 28);
      if      (asi < 8.0f)  oled.print("Normal stiffness");
      else if (asi < 12.0f) oled.print("Mild stiffness");
      else                  oled.print("High stiffness!");
      oled.setCursor(0, 48);
      oled.print(calibrated ? "[Cal OK]" : "[Not calibrated]");
      break;
  }
  oled.display();
}

// ============================================================
// WiFi + HTTP
// ============================================================
void initWiFi() {
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("AP IP: "); Serial.println(WiFi.softAPIP());
}

void initHTTP() {
  http.on("/", HTTP_GET, []() {
    http.send_P(200, "text/html", INDEX_HTML);
  });

  http.on("/calibrate", HTTP_POST, []() {
    if (!http.hasArg("sbp") || !http.hasArg("dbp")) {
      http.send(400, "application/json", "{\"error\":\"Missing sbp or dbp\"}");
      return;
    }
    if (pttMs < 10.0f) {
      http.send(400, "application/json", "{\"error\":\"No PTT yet — place fingers and wait 5s\"}");
      return;
    }
    calSBP     = http.arg("sbp").toFloat();
    calDBP     = http.arg("dbp").toFloat();
    calPTT     = pttMs;
    calibrated = true;
    Serial.printf("[CAL] SBP=%.0f DBP=%.0f PTT=%.1fms\n", calSBP, calDBP, calPTT);
    char resp[80];
    snprintf(resp, sizeof(resp),
             "{\"ok\":true,\"ptt\":%.1f,\"sbp\":%.0f}", calPTT, calSBP);
    http.send(200, "application/json", resp);
  });

  http.on("/data", HTTP_GET, []() {
    StaticJsonDocument<256> d;
    d["sbp"] = (int)sbp;   d["dbp"] = (int)dbp;
    d["hr"]  = (int)hr;    d["ptt"] = (int)pttMs;
    d["pwv"] = pwv;        d["asi"] = asi;
    d["mot"] = motion;     d["cal"] = calibrated;
    d["fin"] = fingerOn;
    char buf[256];
    serializeJson(d, buf);
    http.send(200, "application/json", buf);
  });

  http.begin();
}

// ============================================================
// WEBSOCKET
// ============================================================
void initWS() {
  ws.begin();
  ws.onEvent([](uint8_t n, WStype_t t, uint8_t* p, size_t l) {
    if      (t == WStype_CONNECTED)    { wsClient = true; }
    else if (t == WStype_DISCONNECTED) { wsClient = false; }
  });
}

void pushWS(float w, float f) {
  // FIX 5 — use char buffer + sprintf instead of String() objects
  // String() on every call causes heap fragmentation at 20 Hz
  char buf[220];
  snprintf(buf, sizeof(buf),
    "{\"w\":%.2f,\"f\":%.2f,\"sbp\":%d,\"dbp\":%d,"
    "\"hr\":%d,\"ptt\":%d,\"pwv\":%.2f,\"asi\":%.2f,"
    "\"mot\":%d,\"cal\":%d,\"fin\":%d}",
    w, f,
    (int)sbp, (int)dbp, (int)hr, (int)pttMs,
    pwv, asi,
    motion ? 1 : 0,
    calibrated ? 1 : 0,
    fingerOn ? 1 : 0);
  ws.broadcastTXT(buf);
}
