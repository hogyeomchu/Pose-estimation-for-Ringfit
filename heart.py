# import serial
import vlc
import time


# # 아두이노와 연결 설정
# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)


# def read_data():
#     heart_data = arduino.readline().decode('utf-8').strip()
#     return heart_data

# while True:
#     value = read_data()
#     if value:
#         print(f"센서 값: {value}")



# VLC 인스턴스 생성
player = vlc.MediaPlayer("C:/Users/skydk/Desktop/Pose-estimation-for-Ringfit/music.mp3")

# 배속 설정 (1.5배속)
player.set_rate(1.5)

# 음악 재생
player.play()

# 음악이 끝날 때까지 대기
time.sleep(100000)  # 음악 길이에 맞게 조정

# 음악 종료
player.stop()













