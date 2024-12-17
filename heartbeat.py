import Jetson.GPIO as GPIO
import time
from collections import deque

# GPIO 핀 설정
HEARTBEAT_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HEARTBEAT_PIN, GPIO.IN)

pulse_intervals = []
bpm_history = deque(maxlen=10)
last_pulse_time = 0

def heartbeat_callback(channel):
    """
    GPIO 인터럽트 콜백 함수. 펄스 간격 기록.
    """
    global pulse_intervals, last_pulse_time
    current_time = time.time()
    interval = current_time - last_pulse_time

    if 0.3 < interval < 2.0:  # 최소/최대 간격 설정
        pulse_intervals.append(current_time)
        last_pulse_time = current_time
        print(f"Pulse detected at {current_time:.2f} (Interval: {interval:.2f} sec)")
    else:
        print(f"Ignored pulse at {current_time:.2f} (Interval: {interval:.2f} sec)")

def calculate_bpm():
    """
    기록된 펄스 간격을 기반으로 BPM 계산.
    """
    global pulse_intervals
    if len(pulse_intervals) > 1:
        intervals = [pulse_intervals[i] - pulse_intervals[i-1] for i in range(1, len(pulse_intervals))]
        avg_interval = sum(intervals) / len(intervals)
        bpm = 60 / avg_interval
        return bpm
    return 0

def smooth_bpm(new_bpm):
    """
    스무딩된 BPM 계산 (Moving Average).
    """
    bpm_history.append(new_bpm)
    return sum(bpm_history) / len(bpm_history)

# GPIO 이벤트 감지 (디바운싱 추가)
GPIO.add_event_detect(HEARTBEAT_PIN, GPIO.RISING, callback=heartbeat_callback, bouncetime=300)

try:
    while True:
        time.sleep(2)  # 2초 간격으로 BPM 계산
        bpm = calculate_bpm()
        if bpm > 0:
            bpm = smooth_bpm(bpm)
            print(f"스무딩된 심박수: {bpm:.2f} BPM")
        else:
            print("펄스 감지 중...")

        # 10초마다 간격 리스트 정리
        if len(pulse_intervals) > 1 and time.time() - pulse_intervals[0] > 10:
            pulse_intervals = pulse_intervals[-1:]  # 마지막 값만 유지

except KeyboardInterrupt:
    print("프로그램 종료.")

finally:
    GPIO.cleanup()
