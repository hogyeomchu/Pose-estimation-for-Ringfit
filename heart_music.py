import vlc
import time
import serial
import Jetson.GPIO as GPIO


def heartbeat_callback(channel):
    global pulse_intervals, last_pulse_time
    current_time = time.time()
    interval = current_time - last_pulse_time
    
    if interval > 0:  # 최소 간격 조건 제거
        pulse_intervals.append(current_time)
        last_pulse_time = current_time
        print(f"Pulse detected at {current_time:.2f} (Interval: {interval:.2f} sec)")


HEARTBEAT_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HEARTBEAT_PIN, GPIO.IN)
GPIO.add_event_detect(HEARTBEAT_PIN, GPIO.RISING, callback= heartbeat_callback)


pulse_intervals = []
last_pulse_time = 0

def music(beat):
    player = vlc.MediaPlayer("./Pose-estimation-for-Ringfit/music.mp3")  # nvidia@nvidia-desktop:~/Pose-estimation-for-Ringfit
    beep = vlc.MediaPlayer("./Pose-estimation-for-Ringfit/beep_sound.mp3")
    player.set_rate(1)
    player.play()
    
    if beat > 100 :
        beep.set_rate(1.2)
        beep.play()
        time.sleep(5)
        while True:
            if beat < 80:
                break
    elif beat > 90:
        player.set_rate(1.25)
    
   


def music_heart():
    try: 
        while True:
            if len(pulse_intervals) > 1:
                intervals = [pulse_intervals[i] - pulse_intervals[i-1] for i in range(1, len(pulse_intervals))]
                avg_interval = sum(intervals) / len(intervals)
                bpm = 60 / avg_interval
                print(f"Current BPM: {bpm:.2f}")        #### 이거 put text로 바꾸자 
                music(bpm)
        
    except KeyboardInterrupt:
        print("프로그램 종료.")
    finally:
        GPIO.cleanup()



        

    
    


def main():
    music_heart()
    


if __name__ == "__main__":
    main()





