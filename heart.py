import serial

# 아두이노와 연결 설정
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)


def read_data():
    heart_data = arduino.readline().decode('utf-8').strip()
    return heart_data

while True:
    value = read_data()
    if value:
        print(f"센서 값: {value}")