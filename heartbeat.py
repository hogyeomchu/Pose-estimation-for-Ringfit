from collections import deque
import Jetson.GPIO as GPIO
import time

# GPIO 핀 설정
HEARTBEAT_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HEARTBEAT_PIN, GPIO.IN)

# 심박수 계산 변수
pulse_count = 0
start_time = time.time()
pulse_history = deque(maxlen=10)

def heartbeat_callback(channel):
    global pulse_count
    pulse_count += 1

def smooth_bpm(new_bpm):
    pulse_history.append(new_bpm)
    return sum(pulse_history) / len(pulse_history)

# GPIO Interrupt 설정
GPIO.add_event_detect(HEARTBEAT_PIN, GPIO.RISING, callback=heartbeat_callback)

try:
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            if pulse_count > 0:
                bpm = (pulse_count / elapsed_time) * 60
                bpm = smooth_bpm(bpm)
                print(f"스무딩된 심박수: {bpm:.2f} BPM")
            else:
                print("펄스 신호를 감지하지 못했습니다.")

            pulse_count = 0
            start_time = time.time()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("프로그램 종료.")
finally:
    GPIO.cleanup()
