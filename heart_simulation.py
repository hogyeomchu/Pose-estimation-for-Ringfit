import serial
import time
import threading

# 시리얼 포트와 보드레이트 설정
SERIAL_PORT = "/dev/ttyTHS1"  # Jetson의 UART 포트 (TX, RX 핀)
BAUD_RATE = 115200  # 아두이노와 동일한 보드레이트 설정

# 전역 변수
heartbeat_data = None  # 심박수 데이터 저장
running = True  # 프로그램 상태 플래그

# 시리얼 초기화
def init_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial port {SERIAL_PORT} opened successfully.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

# 심박수 데이터를 읽는 스레드
def read_heartbeat(ser):
    global heartbeat_data, running
    while running:
        if ser.in_waiting > 0:  # 수신된 데이터가 있는 경우
            try:
                line = ser.readline().decode('utf-8').strip()  # 데이터 읽기
                if line.isdigit():  # 심박수 데이터가 숫자인지 확인
                    heartbeat_data = int(line)
                    print(f"Heartbeat: {heartbeat_data} BPM")
            except Exception as e:
                print(f"Error reading data: {e}")
        time.sleep(0.1)  # CPU 과부하 방지를 위한 짧은 대기

# 메인 프로그램
def main():
    global running

    # 시리얼 초기화
    ser = init_serial()
    if ser is None:
        print("Failed to initialize serial connection.")
        return

    # 심박수 읽기 스레드 시작
    thread = threading.Thread(target=read_heartbeat, args=(ser,))
    thread.start()

    try:
        print("Press Ctrl+C to exit.")
        while running:
            # 메인 루프: 심박수 데이터를 활용한 추가 작업 수행
            if heartbeat_data is not None:
                if heartbeat_data > 100:  # 예: 심박수가 100을 초과하면 경고 출력
                    print("Warning: High heartbeat detected!")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting program.")
        running = False
        thread.join()  # 스레드 종료 대기
        ser.close()  # 시리얼 포트 닫기

if __name__ == "__main__":
    main()
