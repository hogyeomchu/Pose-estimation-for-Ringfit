import sched
import time 
import cv2 
import Jetson.GPIO as GPIO
from datetime import datetime

# GPIO 핀 번호
INPUT_PIN = 12  # 이벤트를 감지할 핀 번호

# GPIO 설정
GPIO.setmode(GPIO.BOARD)  # BOARD 모드 사용
GPIO.setup(INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # 입력 모드 설정

# 타이머 상태 변수
start_time = None
timer_duration = 5  # 타이머 지속 시간 (초)

def interrupt_handler(channel):
    """
    Interrupt가 발생했을 때 실행되는 핸들러 함수.
    """
    global start_time
    if start_time is None:
        start_time = time.time()
        print("타이머 시작!")
    else:
        elapsed_time = time.time() - start_time
        print(f"타이머 종료! 경과 시간: {elapsed_time:.2f}초")
        start_time = None

# GPIO Interrupt 설정
GPIO.add_event_detect(INPUT_PIN, GPIO.RISING, callback=interrupt_handler, bouncetime=200)

print("Interrupt 기반 타이머 준비 완료. 입력 신호를 기다리는 중...")

try:
    while True:
        if start_time:
            elapsed_time = time.time() - start_time
            if elapsed_time >= timer_duration:
                print(f"타이머 완료: {timer_duration}초 경과")
                start_time = None
        time.sleep(0.1)  # CPU 사용량 감소를 위해 대기

except KeyboardInterrupt:
    print("프로그램 종료.")

finally:
    GPIO.cleanup()  # GPIO 설정 정리
