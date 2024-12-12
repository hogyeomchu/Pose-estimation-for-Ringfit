import cv2
import os
import datetime
from ultralytics import YOLO

# 저장 폴더 생성
output_folder = "saved_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# YOLOv8 Pose 모델 로드
model = YOLO("yolov8s-pose.pt")  # 사전 학습된 YOLOv8 Pose 모델

# 카메라 열기
cap = cv2.VideoCapture(0)  # 기본 카메라 사용
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 바운딩 박스 설정
bbox_x, bbox_y, bbox_width, bbox_height = 540, 700, 200, 75  # 기본 바운딩 박스 값
print("Q를 눌러 프로그램을 종료하고, S를 눌러 프레임과 스켈레톤 좌표를 저장하세요.")

# 스켈레톤 좌표 저장용 리스트
skeleton_data = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 800))
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLOv8 Pose 추론 실행
    results = model(frame)
    keypoints = results[0].keypoints  # 스켈레톤 키포인트 추출

    # 바운딩 박스 그리기
    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (255, 0, 0), 2)

    # 스켈레톤 시각화
    if keypoints is not None:
        for person_keypoints in keypoints.data:
            for x, y, conf in person_keypoints:  # 각 키포인트 좌표
                if conf > 0.5:  # 신뢰도 기준
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 화면 출력
    cv2.imshow("Foot Detection", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 프로그램 종료
        break
    elif key == ord('s'):  # 프레임과 스켈레톤 좌표 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"프레임 저장: {filename}")

        # 스켈레톤 좌표 저장
        if keypoints is not None:
            for person_keypoints in keypoints.data:
                skeleton = person_keypoints.cpu().numpy().tolist()  # NumPy 배열을 리스트로 변환
                skeleton_data.append({
                    "timestamp": timestamp,
                    "skeleton": skeleton
                })
                print(f"스켈레톤 저장: {skeleton}")

# 자원 해제
cap.release()
cv2.destroyAllWindows()

# 스켈레톤 데이터 저장
output_skeleton_file = os.path.join(output_folder, "skeleton_data.txt")
with open(output_skeleton_file, "w") as f:
    for entry in skeleton_data:
        f.write(f"Timestamp: {entry['timestamp']}, Skeleton: {entry['skeleton']}\n")
print(f"스켈레톤 데이터 저장 완료: {output_skeleton_file}")