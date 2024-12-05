import cv2
import os
import datetime

# 저장 폴더 생성
output_folder = "saved_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 카메라 열기
cap = cv2.VideoCapture(0)  # 기본 카메라 사용
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 바운딩 박스 설정
bbox_x, bbox_y, bbox_width, bbox_height = 170, 400, 300, 50  # 기본 바운딩 박스 값

print("Q를 눌러 프로그램을 종료하고, S를 눌러 프레임을 저장하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 바운딩 박스 그리기
    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Foot Detection", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 프로그램 종료
        break
    elif key == ord('s'):  # 프레임 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"프레임 저장: {filename}")

# 자원 해제
cap.release()
cv2.destroyAllWindows()
