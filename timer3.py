import Jetson.GPIO as GPIO
import threading
import time

# GPIO 핀 설정
interrupt_pin = 16  # GPIO 핀 번호 (BOARD 모드 기준)
boundary = 100  # 임계값

# 전역 변수
a = 90  # 센서 데이터 (실시간으로 변경된다고 가정)
timer_running = False
timer_event = threading.Event()
#dddd
# GPIO 초기화
def setup_gpio():
    GPIO.setmode(GPIO.BOARD)  # GPIO 핀 번호 설정 방식
    GPIO.setup(interrupt_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # 핀 설정 (입력 핀으로 설정)

# 타이머 종료 함수
def stop_timer():
    global timer_running
    timer_event.set()  # 이벤트를 설정하여 타이머를 중단
    timer_running = False
    print("타이머 중단 및 초기화")

# # 타이머 중단 함수 
# def interrupt_and_reset_timer():
#     global timer_running
#     if timer_running:  # 타이머가 실행 중일 경우
#         print("남은 시간이 1초일 때 타이머를 중단하고 초기화합니다.")
#         stop_timer()

# 타이머 시작 함수
def start_timer(duration):
    global timer_running
    if timer_running:  # 이미 타이머가 실행 중이면 중단
        stop_timer()

    timer_running = True
    timer_event.clear()  # 이벤트 초기화
    print(f"타이머 시작! {duration}초")

    def timer_task():
        nonlocal duration
        while duration > 0:
            if timer_event.is_set():  # 이벤트가 설정되면 중단
                return
            print(f"남은 시간: {duration}초")
            time.sleep(1)
            duration -= 1
            if duration == 1:
                stop_timer()


        print("타이머 종료!")
        timer_running = False

    threading.Thread(target=timer_task, daemon=True).start()

# GPIO 핀 상태 업데이트
def update_gpio_state():
    global a, timer_running
    if a < boundary:  # a가 임계값보다 작을 경우
        if not timer_running:  # 타이머가 실행 중이 아니면 시작
            start_timer(3)
    else:
        if timer_running:  # a가 임계값보다 커지면 타이머 중단
            stop_timer()

# 센서 데이터 업데이트 시뮬레이션
def sensor_update_simulation():
    global a
    while True:
        try:
            a = int(input("a 값 입력 (0~200): "))
            update_gpio_state()
        except ValueError:
            print("유효한 숫자를 입력하세요.")
        time.sleep(0.1)

    


# 메인 실행
try:
    setup_gpio()
    print("프로그램 실행 중... 센서 데이터 입력 대기")
    threading.Thread(target=sensor_update_simulation, daemon=True).start()  # 센서 업데이트 스레드 실행
    while True:
        time.sleep(1)  # 메인 루프 유지
except KeyboardInterrupt:
    print("프로그램 종료")
finally:
    GPIO.cleanup()
    print("GPIO 정리 완료")
