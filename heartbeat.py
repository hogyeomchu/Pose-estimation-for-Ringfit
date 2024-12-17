import Jetson.GPIO as GPIO
import time

# GPIO 핀 설정
HEARTBEAT_PIN = 12  # 심박수 센서를 연결한 GPIO 핀 번호
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HEARTBEAT_PIN, GPIO.IN)

# 심박수 계산 변수
pulse_count = 0
start_time = time.time()

def heartbeat_callback(channel):
    """
    심박수 센서의 펄스 신호를 감지하는 콜백 함수.
    """
    global pulse_count
    pulse_count += 1  # 펄스 발생 시마다 증가

# GPIO 이벤트 감지 설정 (RISING 엣지에서 신호 발생)
GPIO.add_event_detect(HEARTBEAT_PIN, GPIO.RISING, callback=heartbeat_callback)

try:
    while True:
        # 10초 간격으로 심박수 계산
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:  # 10초 동안의 펄스 카운트
            bpm = (pulse_count / elapsed_time) * 60  # 심박수 계산
            print(f"심박수: {bpm:.2f} BPM")
            
            # 초기화
            pulse_count = 0
            start_time = time.time()

        time.sleep(0.1)  # CPU 사용량 감소를 위해 대기

except KeyboardInterrupt:
    print("프로그램 종료.")

finally:
    GPIO.cleanup()  # GPIO 정리
