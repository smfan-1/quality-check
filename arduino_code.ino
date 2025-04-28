#include <Servo.h>

const int motorPin = 8;      // DC motor control pin (ON/OFF)
const int servoPin = 9;      // Servo signal pin
Servo myServo;

void setup() {
  pinMode(motorPin, OUTPUT);
  myServo.attach(servoPin);
  
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    

    
    if (cmd == 'D') {
      // Defective bottle detected:
      // Stop the conveyor
      digitalWrite(motorPin, LOW);
      
      // Run servo motor sequence to handle defective bottle (e.g., divert it)
      digitalWrite(motorPin, HIGH);
      delay(7000);  // Adjust the delay as per your conveyor belt speed
      digitalWrite(motorPin, LOW);
      myServo.write(90);
      delay(1000);
      myServo.write(180);
      delay(1000);

      myServo.write(90);
      delay(1000);

    } 
    else if (cmd == 'N') {
      // Non-defective bottle: let it pass.
      // Start conveyor for a short time to move the bottle
      digitalWrite(motorPin, HIGH);
      delay(7000);  // Adjust the delay as per your conveyor belt speed
      digitalWrite(motorPin, LOW);
    }
  }
}
