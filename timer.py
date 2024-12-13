import sched
import time 
import cv2 

# 전역 변수 설정
timer_running = False  # 타이머 실행 여부
remaining_time          # 초기 타이머 시간 설정 (초 단위)
scheduler = sched.scheduler(time.time, time.sleep)  # sched 스케줄러 객체 생성



# 타이머 감소 함수
def countdown_timer():
    global remaining_time, timer_running
    if remaining_time > 0:
        remaining_time -= 1  # 남은 시간 1초 감소
        scheduler.enter(1, 1, countdown_timer)  # 1초 후에 다시 실행 (delay, priority, 호출 함수)
    else:
        # timer =0 이 됐을 떄 소리가 나게 한다. 
        timer_running = False  # 타이머 종료


# 타이머 시작 함수
def start_timer(duration):
    global remaining_time, timer_running
    remaining_time = duration  # 초기화
    if not timer_running:
        timer_running = True
        scheduler.enter(1, 1, countdown_timer)  # 타이머 시작





# 화면에 텍스트 출력
def put_text(frame, exercise, count, fps, heart_data, redio):
    global remaining_time

    # 텍스트 배경 그리기
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(250 * redio)),
        (55, 104, 0), -1
    )

    # 타이머 상태 표시
    cv2.putText(
        frame, f'Timer: {remaining_time}', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(100 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio), int(150 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame, f'Heart: {heart_data}', (int(30 * redio), int(200 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame, f'TIME: {remaining_time}', (int(30 * redio), int(250 * redio)), 0, 0.9 * redio,
        (255, 0, 0), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )







# 메인 루프
def main():
    global remaining_time
    cap = cv2.VideoCapture(0)  # 카메라 시작
    redio = 1.0  # 스케일링 비율
    exercise = "Pushup"
    count = 0
    fps = 30
    heart_data = 72

    start_timer(3)  # 3초 타이머 시작 (맨 처음 시작할 때는 선 자세에서 bounding box안에 keypoint가 들어오는 것을 확인 후 시작)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 조건이 만족되면 타이머를 시작 (예시 조건: count > 5)
        if count > 5 and not timer_running:
            start_timer(3)

        # 텍스트 출력 함수 호출
        put_text(frame, exercise, count, fps, heart_data, redio)

        cv2.imshow("Timer Example", frame)  # 화면 표시
        scheduler.run(blocking=False)  # 스케줄러 실행 (타이머 감소)

        # 종료 조건 (ESC 키)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






if state == "ready": 
    start_timer(3)



if state == "start": 
    if (remaining_time == 3  and error < 100):
        countdown_timer()

############################ GPIO interrupt 기반 타이머 ############################

import Jetson.GPIO as GPIO
import time
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
