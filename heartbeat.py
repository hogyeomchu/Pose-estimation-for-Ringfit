import Jetson.GPIO as GPIO
import time

HEARTBEAT_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HEARTBEAT_PIN, GPIO.IN)

pulse_intervals = []
last_pulse_time = 0

def heartbeat_callback(channel):
    global pulse_intervals, last_pulse_time
    current_time = time.time()
    interval = current_time - last_pulse_time
    
    if interval > 0.2:  # 최소 간격 조건 제거
        pulse_intervals.append(current_time)
        last_pulse_time = current_time
        print(f"Pulse detected at {current_time:.2f} (Interval: {interval:.2f} sec)")

GPIO.add_event_detect(HEARTBEAT_PIN, GPIO.RISING, callback=heartbeat_callback)

try:
    while True:
        if len(pulse_intervals) > 1:
            intervals = [pulse_intervals[i] - pulse_intervals[i-1] for i in range(1, len(pulse_intervals))]
            avg_interval = sum(intervals) / len(intervals)
            bpm = 60 / avg_interval
            print(f"Current BPM: {bpm:.2f}")
        
        time.sleep(2)  # 2초 간격으로 BPM 출력
except KeyboardInterrupt:
    print("프로그램 종료.")
finally:
    GPIO.cleanup()
