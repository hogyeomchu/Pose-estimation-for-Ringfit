import vlc
import time
import serial

# try:
#     arduino = serial.Serial(port='/dev/ttyTHS1', baudrate=9600, timeout=1)  # jetson nano 기본 uART pin: ttyTHS*
#     print("아두이노와 연결되었습니다.")
# except serial.SerialException:
#     print("아두이노 연결 실패. 포트를 확인하세요.")
#     exit()



# def read_data():
#     try:
#         # 시리얼 데이터 읽기
#         data = arduino.readline().decode('utf-8').strip()  # 데이터 읽기 및 디코딩
#         if data:
#             heart_data = int(data)  # 정수로 변환
#             return heart_data
#         else:
#             print("경고: 데이터를 읽을 수 없습니다.")
#             return None
#     except ValueError:
#         print("경고: 데이터 형식이 잘못되었습니다.")
#         return None
#     except Exception as e:
#         print(f"예기치 않은 오류 발생: {e}")
#         return None


def music():
    player = vlc.MediaPlayer("nvidia@nvidia-desktop:~/Pose-estimation-for-Ringfit/music.mp3")  # VLC 인스턴스 생성
    player.set_rate(1)
    player.play()
    time.sleep(100000)  # 음악 길이에 맞게 조정
    # if(heart_data > 100):
    #         player.set_rate(1.5)
    # elif (heart_data > 90):
    #         player.set_rate(1.2)
    # else:
            




# 음악이 끝날 때까지 대기


# 음악 종료
# player.stop()










