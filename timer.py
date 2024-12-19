import Jetson.GPIO as GPIO
import threading
import time

# GPIO 핀 설정
interrupt_pin = 16  # GPIO 핀 번호 (BOARD 모드 기준)
boundary = 100  # 임계값

# 전역 변수
a = 90  # 센서 데이터 (실시간으로 변경된다고 가정)
timer_state = {
    "running": False,
    "over": False,
}
timer_event = threading.Event()
lock = threading.Lock()  # 동기화를 위한 Lock 객체

# GPIO 초기화
def setup_gpio():
    GPIO.setmode(GPIO.BOARD)  # GPIO 핀 번호 설정 방식
    GPIO.setup(interrupt_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # 핀 설정 (입력 핀으로 설정)

# 타이머 종료 함수
def stop_timer():
    with lock:
        timer_event.set()  # 이벤트를 설정하여 타이머를 중단
        timer_state["running"] = False
        print("타이머 중단 및 초기화")

# 타이머 시작 함수
def start_timer(duration):
    with lock:
        if timer_state["running"]:  # 이미 타이머가 실행 중이면 중단
            stop_timer()

        timer_state["running"] = True
        timer_state["over"] = False
        timer_event.clear()  # 이벤트 초기화
        print("타이머 시작!")

    def timer_task():
        nonlocal duration
        while duration > 0:
            if timer_event.is_set():  # 이벤트가 설정되면 중단
                return
            print(f"남은 시간: {duration}초")
            time.sleep(1)
            duration -= 1

        with lock:
            timer_state["running"] = False
            timer_state["over"] = True
            print("타이머 종료!")

    threading.Thread(target=timer_task, daemon=True).start()

# 상태 확인 함수
def is_timer_running():
    with lock:
        return timer_state["running"]

def is_timer_over():
    with lock:
        return timer_state["over"]

# 센서 데이터 업데이트 및 타이머 제어 함수
def update_gpio_state(new_a):
    global a
    a = new_a
    if a < boundary:  # a가 임계값보다 작을 경우
        if not is_timer_running():  # 타이머가 실행 중이 아니면 시작
            start_timer(3)
    else:
        if is_timer_running():  # a가 임계값보다 커지면 타이머 중단
            stop_timer()

# GPIO 정리 함수
def cleanup_gpio():
    GPIO.cleanup()
    print("GPIO 정리 완료")
