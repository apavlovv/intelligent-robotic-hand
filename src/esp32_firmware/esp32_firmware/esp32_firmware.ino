#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define I2C_SDA 8
#define I2C_SCL 9

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

struct ServoConfig {
  int minPWM;
  int midPWM;
  int maxPWM;
};

struct ServoState {
  float currentPWM;
  int targetPWM;
};

ServoConfig servos[16] = {
  {144, 324, 500}, //(Мізинець,кінчик)
  {166, 347, 500}, //(Мізинець,середина)
  {177, 366, 500}, //(Мізинець,основа)
  {188, 355, 500}, //(Безіменний,кінчик)
  {188, 355, 500}, //(Безіменний,середина)
  {188, 366, 500}, //(Безіменний,основа)
  {200, 353, 500}, //(Великий,основа)
  {196, 320, 400}, //(Великий,середина90)
  {200, 355, 500}, //(Великий,середина)
  {177, 340, 500}, //(Великий,кінчик)
  {155, 326, 500}, //(Середній,основа)
  {144, 326, 500}, //(Середній,кінчик)
  {144, 326, 500}, //(Середній,середина)
  {144, 326, 500}, //(Вказівний,кінчик)
  {133, 333, 500}, //(Вказівний,середина)
  {144, 333, 500}  //(Вказівний,основа)
};

ServoState states[16];

const float EMA_ALPHA = 0.4; 
const int FAST_SERVO = 6;            
unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5); 
  Wire.begin(I2C_SDA, I2C_SCL);
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);

  for (int i = 0; i < 16; i++) {
    states[i].currentPWM = -1; 
    states[i].targetPWM = -1;
  }
  
  delay(10);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.peek();
    
    if (c == 'A' || c == 'P') {
      char cmdType = Serial.read(); 
      Serial.read();                
      int channel = Serial.parseInt();
      Serial.read();                
      int value = Serial.parseInt();
      
      while (Serial.available() > 0) {
        char dump = Serial.read();
        if (dump == '\n') break;
      }
      
      if (channel >= 0 && channel <= 15) {
        if (cmdType == 'A' && value >= 0 && value <= 180) {
          int target = (value <= 90) ? 
                       map(value, 0, 90, servos[channel].minPWM, servos[channel].midPWM) : 
                       map(value, 90, 180, servos[channel].midPWM, servos[channel].maxPWM);
          
          states[channel].targetPWM = target;
          
          if (states[channel].currentPWM == -1) {
            states[channel].currentPWM = target;
            pwm.setPWM(channel, 0, target);
          }
        } else if (cmdType == 'P') {
          states[channel].currentPWM = value;
          states[channel].targetPWM = value;
          pwm.setPWM(channel, 0, value);
        }
      }
    } else {
      Serial.read(); 
    }
  }

  unsigned long currentMillis = millis();
  
  if (currentMillis - lastUpdate >= 15) {
    lastUpdate = currentMillis;
    
    for (int i = 0; i < 16; i++) {
      if (states[i].targetPWM != -1 && states[i].currentPWM != states[i].targetPWM) {
        
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