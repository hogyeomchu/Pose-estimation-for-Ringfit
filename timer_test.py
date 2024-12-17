import sched
import time 
import cv2 

# 전역 변수 설정
timer_running = False  # 타이머 실행 여부
remaining_time = 0         # 초기 타이머 시간 설정 (초 단위)
scheduler = sched.scheduler(time.time, time.sleep)  # sched 스케줄러 객체 생성



# 타이머 감소 함수
def countdown_timer():
    global remaining_time, timer_running
    if remaining_time > 0:
        print("time left: ", remaining_time)
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











# 메인 루프
def main():
    global remaining_time
    start_timer(3)
    scheduler.run()  # 스케줄러를 실행하여 타이머를 동작시킴


    
if __name__ == "__main__":
    main()



