import cv2

# 고정된 스켈레톤 좌표 (예: 17개의 키포인트)
# (x, y) 좌표 리스트를 정의합니다.
fixed_skeleton = [
    (200, 100),  # Nose
    (220, 80),   # Left Eye
    (180, 80),   # Right Eye
    (240, 100),  # Left Ear
    (160, 100),  # Right Ear
    (200, 150),  # Left Shoulder
    (200, 150),  # Right Shoulder
    (240, 200),  # Left Elbow
    (160, 200),  # Right Elbow
    (260, 250),  # Left Wrist
    (140, 250),  # Right Wrist
    (220, 250),  # Left Hip
    (180, 250),  # Right Hip
    (240, 300),  # Left Knee
    (160, 300),  # Right Knee
    (260, 350),  # Left Ankle
    (140, 350),  # Right Ankle
]

# 스켈레톤 연결선 정의 (COCO 포맷 예시)
# (start_point_index, end_point_index)
skeleton_connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
    (5, 6), (5, 7), (6, 8),          # Shoulders to Elbows
    (7, 9), (8, 10),                 # Elbows to Wrists
    (11, 12), (11, 13), (12, 14),    # Hips to Knees
    (13, 15), (14, 16)               # Knees to Ankles
]

# 카메라 열기
cap = cv2.VideoCapture(0)  # 기본 카메라 사용
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("Q를 눌러 프로그램을 종료하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 고정된 스켈레톤 키포인트 그리기
    for x, y in fixed_skeleton:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # 빨간 점

    # 고정된 스켈레톤 연결선 그리기
    for start_idx, end_idx in skeleton_connections:
        start_point = fixed_skeleton[start_idx]
        end_point = fixed_skeleton[end_idx]
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # 파란 선

    # 화면 출력
    cv2.imshow("Fixed Skeleton Overlay", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 프로그램 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
