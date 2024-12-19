# test_timer.py
import timer3 as timer  # 위의 파일이 'timer.py'로 저장되어 있다고 가정
import time

def main():
    # 타이머가 시작되지 않았음을 확인
    print("타이머 실행 여부:", timer.is_timer_running())
    print("타이머 종료 여부:", timer.is_timer_over())
    
    # 타이머 시작
    print("\n타이머를 3초 동안 시작합니다.")
    timer.start_timer(3)

    # 타이머 진행 확인
    while timer.is_timer_running():
        print("타이머 실행 중...")
        print(timer.is_timer_over())
        time.sleep(1)

    # 타이머 종료 여부 확인
    print("\n타이머 실행 여부:", timer.is_timer_running())
    print("타이머 종료 여부:", timer.is_timer_over())

    # 타이머 종료 후 다시 시작
    print("\n타이머를 5초 동안 다시 시작합니다.")
    timer.start_timer(5)

    # 타이머 진행 확인
    while timer.is_timer_running():
        print("타이머 실행 중...")
        time.sleep(1)

    # 타이머 종료 여부 확인
    print("\n타이머 실행 여부:", timer.is_timer_running())
    print("타이머 종료 여부:", timer.is_timer_over())

if __name__ == "__main__":
    main()
