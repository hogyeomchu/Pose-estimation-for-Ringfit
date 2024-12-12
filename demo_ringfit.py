# 타이머 관련 변수 설정
timer_running = False  # 타이머 실행 여부
remaining_time = 3  # 초기 타이머 시간 설정 (초 단위)
scheduler = sched.scheduler(time.time, time.sleep)  # sched 스케줄러 객체 생성

# 타이머 감소 함수
def countdown_timer():
    global remaining_time, timer_running
    if remaining_time > 0:
        remaining_time -= 1  # 남은 시간 1초 감소
        scheduler.enter(1, 1, countdown_timer)  # 1초 후에 다시 실행 (delay, priority, 호출 함수)
    else:
        timer_running = False  # 타이머 종료

# 타이머 시작 함수
def start_timer(duration):
    global remaining_time, timer_running
    remaining_time = duration  # 초기화
    if not timer_running:
        timer_running = True
        scheduler.enter(1, 1, countdown_timer)  # 타이머 시작

# FSM 상태 클래스 정의
class State:
    def __init__(self, delay, nstate1=None, nstate2=None, nstate3=None):
        self.delay = delay       # 상태의 지연 시간
        self.nstate1 = nstate1   # 다음 상태 1
        self.nstate2 = nstate2   # 다음 상태 2
        self.nstate3 = nstate3   # 다음 상태 3

# FSM 클래스 정의
class FSM:
    def __init__(self):
        # 상태 초기화
        self.states = {  
            "ready": State(delay=5, nstate1="start", nstate2="start", nstate3=None),
            "start": State(delay=3, nstate1="redo", nstate2="redo", nstate3="ready"),
            "redo": State(delay=3, nstate1="start", nstate2="finish", nstate3="ready"),
            "finish": State(delay=0, nstate1="ready", nstate2=None, nstate3=None),
        }
        self.current_state = "ready"  # 초기 상태 설정

    def transition(self, condition1, condition2, condition3):   #  timer == 0 / timer == 0 & count==10 / Key interrupt
        # 현재 상태의 객체 가져오기
        state_obj = self.states[self.current_state]

        # 상태의 delay 값을 타이머 시작 함수에 전달하여 타이머 초기화
        start_timer(state_obj.delay)

        # 조건에 따라 다음 상태 결정
        if condition1 == 1 and state_obj.nstate1:
            self.current_state = state_obj.nstate1
        if condition2 == 1 and state_obj.nstate2:
            self.current_state = state_obj.nstate2
        if condition3 == 1 and state_obj.nstate3:
            self.current_state = state_obj.nstate3
        else:
            print("유효하지 않은 조건입니다.")

# 메인 함수 내의 FSM 및 타이머 적용 부분
def main():
    # FSM 객체 생성
    fsm = FSM()

    # 상태에 맞는 로직 수행
    while True:
        # 상태에 맞는 처리
        if fsm.current_state == "ready":
            # 'ready' 상태에서의 처리
            start_timer(fsm.states[fsm.current_state].delay)
            print("준비 상태입니다.")
            # 예: 타이머가 0이 되면 "start"로 전환
            fsm.transition(condition1=int(remaining_time == 0), condition2=0, condition3=0)

        elif fsm.current_state == "start":
            # 'start' 상태에서의 처리
            start_timer(fsm.states[fsm.current_state].delay)
            print("시작 상태입니다.")
            # 예: 타이머가 0이 되면 "redo"로 전환
            fsm.transition(condition1=int(remaining_time == 0), condition2=0, condition3=0)

        elif fsm.current_state == "redo":
            # 'redo' 상태에서의 처리
            start_timer(fsm.states[fsm.current_state].delay)
            print("재시작 상태입니다.")
            # 예: 타이머가 0이 되면 "finish"로 전환
            fsm.transition(condition1=int(remaining_time == 0), condition2=0, condition3=0)

        elif fsm.current_state == "finish":
            # 'finish' 상태에서의 처리
            start_timer(fsm.states[fsm.current_state].delay)
            print("완료 상태입니다.")
            # 예: 완료 후 "ready"로 돌아가기
            fsm.transition(condition1=int(remaining_time == 0), condition2=0, condition3=0)

        # 조건에 맞게 상태를 변경하거나 종료할 수 있습니다.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
