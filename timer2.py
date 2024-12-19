import Jetson.GPIO as GPIO
import threading
import time

# GPIO 핀 설정
interrupt_pin = 18  # GPIO 핀 번호 (BOARD 모드 기준)
boundary = 100  # 임계값

# 전역 변수
a = 150  # 센서 데이터 (실시간으로 변경된다고 가정)
timer_running = False
timer_event = threading.Event()

# GPIO 초기화
def setup_gpio():
    GPIO.setmode(GPIO.BOARD)  # GPIO 핀 번호 설정 방식
    GPIO.setup(interrupt_pin, GPIO.IN)  # 핀 설정 (입력 핀으로 설정)
    GPIO.add_event_detect(interrupt_pin, GPIO.FALLING, callback=interrupt_handler, bouncetime=300)  # 인터럽트 설정

# 인터럽트 핸들러 함수
def interrupt_handler(channel):
    global timer_running
    if not timer_running:  # 타이머가 실행 중이 아닐 경우에만 실행
        print(f"GPIO 인터럽트 트리거됨! a: {a}")
        start_timer(3)  # 타이머 시작
    else:
        print("타이머 실행 중 - 인터럽트 무시")

# 소프트웨어에서 인터럽트를 시뮬레이션
def simulate_interrupt():
    global a
    if a < boundary:  # 조건에 따라 인터럽트를 시뮬레이션
        print(f"Simulating interrupt for a={a}")
        interrupt_handler(interrupt_pin)  # 직접 핸들러 호출로 인터럽트 시뮬레이션

# 타이머 함수
def start_timer(duration):
    global timer_running
    timer_running = True
    print(f"타이머 시작! {duration}초")
    timer_event.clear()

    def timer_task():
        global timer_running
        time_remaining = duration
        while time_remaining > 0:
            if timer_event.is_set():
                print("타이머 중단됨.")
                timer_running = False
                return
            print(f"타이머 남은 시간: {time_remaining}초")
            time.sleep(1)
            time_remaining -= 1

        print("타이머 완료!")
        timer_running = False

    threading.Thread(target=timer_task, daemon=True).start()

# 타이머 중단 함수
def stop_timer():
    global timer_running
    if timer_running:
        print("타이머 초기화됨!")
        timer_event.set()  # 타이머 종료 이벤트 트리거
        timer_running = False

# 센서 데이터 업데이트 시뮬레이션
def sensor_update_simulation():
    global a
    while True:
        try:
            # 센서 값이 실시간으로 변한다고 가정 (사용자 입력)
            a = int(input("a 값 입력 (0~200): "))
            # 인터럽트 시뮬레이션
            simulate_interrupt()
        except ValueError:
            print("유효한 숫자를 입력하세요!")
        time.sleep(0.1)  # 값 업데이트 주기

# GPIO 정리
def cleanup_gpio():
    GPIO.cleanup()
    print("GPIO 정리 완료")

# 메인 실행
try:
    setup_gpio()
    print("프로그램 실행 중... GPIO 인터럽트 대기")
    threading.Thread(target=sensor_update_simulation, daemon=True).start()  # 센서 업데이트 스레드 실행
    while True:
        time.sleep(1)  # 메인 루프 유지
except KeyboardInterrupt:
    print("프로그램 종료")
finally:
    cleanup_gpio()
