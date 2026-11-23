const int SENSE_PIN = 2;
const int LED_PIN = 13;
const int BUZZER_PIN = 8;

int last = HIGH;

void setup() {
  pinMode(SENSE_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  int now = digitalRead(SENSE_PIN);
  digitalWrite(LED_PIN, now == LOW);
  if (now == LOW && last == HIGH) {
    Serial.println("B");
    tone(BUZZER_PIN, 2000, 150);
  }
  last = now;
  delay(5);
}
