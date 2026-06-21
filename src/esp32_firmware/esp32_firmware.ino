#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>

#define I2C_SDA 8
#define I2C_SCL 9

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
Preferences preferences;

struct ServoConfig {
  int minPWM; 
  int midPWM; 
  int maxPWM; 
};

struct ServoState {
  float currentPWM;
  int targetPWM;
};

// СТАНДАРТНІ ЗНАЧЕННЯ ЗІ СКРІНШОТУ
ServoConfig defaultServos[16] = {
  {135, 324, 500}, // Канал 0
  {166, 348, 500}, // Канал 1
  {177, 348, 500}, // Канал 2
  {188, 355, 500}, // Канал 3
  {188, 328, 500}, // Канал 4
  {130, 315, 500}, // Канал 5  
  {200, 353, 500}, // Канал 6
  {158, 328, 400}, // Канал 7
  {140, 344, 500}, // Канал 8
  {150, 329, 500}, // Канал 9
  {155, 326, 500}, // Канал 10
  {125, 285, 500}, // Канал 11 
  {115, 282, 500}, // Канал 12 
  {144, 326, 500}, // Канал 13
  {133, 333, 500}, // Канал 14
  {144, 333, 500}  // Канал 15
};

ServoConfig servos[16];
ServoState states[16];

const float EMA_ALPHA = 0.4;
const int FAST_SERVO = 6;
unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10); 
  Wire.begin(I2C_SDA, I2C_SCL);

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);

  preferences.begin("bionic_hand", false);
  
  size_t configSize = preferences.getBytesLength("servo_cfg_v3");
  if (configSize == sizeof(servos)) {
    preferences.getBytes("servo_cfg_v3", servos, sizeof(servos));
    Serial.println("INFO: Налаштування завантажено з NVS.");
  } else {
    memcpy(servos, defaultServos, sizeof(servos));
    preferences.putBytes("servo_cfg_v3", servos, sizeof(servos));
    Serial.println("INFO: Стандартні ліміти збережено в пам'ять.");
  }

  for (int i = 0; i < 16; i++) {
    states[i].currentPWM = -1;
    states[i].targetPWM = -1;
  }
  delay(10);
}

int calculatePWM(int channel, int angle) {
  if (angle <= 90) {
    return map(angle, 0, 90, servos[channel].minPWM, servos[channel].midPWM);
  } else {
    return map(angle, 90, 180, servos[channel].midPWM, servos[channel].maxPWM);
  }
}

void disableUnusedServos(uint16_t activeMask) {
  for (int i = 0; i < 16; i++) {
    if ((activeMask & (1 << i)) == 0) {
      if (states[i].targetPWM != -1) {
        pwm.setPWM(i, 0, 0); 
        states[i].targetPWM = -1; 
        states[i].currentPWM = -1; 
      }
    }
  }
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      char cmdType = command.charAt(0);
      
      if (cmdType == 'A') {
        int channel, angle;
        if (sscanf(command.c_str(), "A,%d,%d", &channel, &angle) == 2) {
          if (channel >= 0 && channel < 16 && angle >= 0 && angle <= 180) {
            int target = calculatePWM(channel, angle);
            states[channel].targetPWM = target;
            if (states[channel].currentPWM == -1) {
              states[channel].currentPWM = target;
              pwm.setPWM(channel, 0, target);
            }
          }
        }
      }
      else if (cmdType == 'X') {
        uint16_t mask;
        if (sscanf(command.c_str(), "X,%hu", &mask) == 1) {
          disableUnusedServos(mask);
        }
      }
      else if (cmdType == 'W') {
        uint16_t mask;
        int pwmVal;
        if (sscanf(command.c_str(), "W,%hu,%d", &mask, &pwmVal) == 2) {
          disableUnusedServos(mask);
          for (int i = 0; i < 16; i++) {
            if (mask & (1 << i)) { 
              states[i].targetPWM = pwmVal;
              if (states[i].currentPWM == -1) {
                states[i].currentPWM = pwmVal;
                pwm.setPWM(i, 0, pwmVal);
              }
            }
          }
        }
      }
      else if (cmdType == 'G') {
        uint16_t mask;
        int angle;
        if (sscanf(command.c_str(), "G,%hu,%d", &mask, &angle) == 2) {
          if (angle >= 0 && angle <= 180) {
            disableUnusedServos(mask);
            for (int i = 0; i < 16; i++) {
              if (mask & (1 << i)) {
                int target = calculatePWM(i, angle);
                states[i].targetPWM = target;
                if (states[i].currentPWM == -1) {
                  states[i].currentPWM = target;
                  pwm.setPWM(i, 0, target);
                }
              }
            }
          }
        }
      }
      else if (cmdType == 'C') {
        int channel, minP, midP, maxP;
        if (sscanf(command.c_str(), "C,%d,%d,%d,%d", &channel, &minP, &midP, &maxP) == 4) {
          if (channel >= 0 && channel < 16) {
            servos[channel].minPWM = minP;
            servos[channel].midPWM = midP;
            servos[channel].maxPWM = maxP;
            Serial.printf("OK: Конфіг серво %d оновлено\n", channel);
          }
        }
      }
      else if (cmdType == 'S') {
        preferences.putBytes("servo_cfg_v3", servos, sizeof(servos));
        Serial.println("OK: Всі налаштування збережено у флеш-пам'ять!");
      }
      else if (cmdType == 'R') {
        for (int i = 0; i < 16; i++) {
          Serial.printf("CFG,%d,%d,%d,%d\n", i, servos[i].minPWM, servos[i].midPWM, servos[i].maxPWM);
        }
        Serial.println("OK: Дані відправлено");
      }
    }
  }

  unsigned long currentMillis = millis();
  if (currentMillis - lastUpdate >= 15) {
    lastUpdate = currentMillis;
    for (int i = 0; i < 16; i++) {
      if (states[i].targetPWM != -1 && abs(states[i].currentPWM - states[i].targetPWM) > 0.1) {
        if (i == FAST_SERVO) {
          states[i].currentPWM = states[i].targetPWM;
        } else {
          states[i].currentPWM = (EMA_ALPHA * states[i].targetPWM) + ((1.0 - EMA_ALPHA) * states[i].currentPWM);
          if (abs(states[i].targetPWM - states[i].currentPWM) < 1.0) {
            states[i].currentPWM = states[i].targetPWM;
          }
        }
        pwm.setPWM(i, 0, (int)states[i].currentPWM);
      }
    }
  }
}