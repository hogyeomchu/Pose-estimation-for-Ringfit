import cv2
from ultralytics import YOLO

def extract_keypoints(image_path, model_path='yolov8s-pose.pt'):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 이미지 로드
    image = cv2.imread(image_path)

    # YOLO 추론 실행
    results = model(image)

    # 키포인트 추출
    keypoints = results[0].keypoints  # shape: (N, 17, 3) where N = number of persons

    if keypoints is not None:
        # 각 사람의 관절 위치를 출력
        for i, person_keypoints in enumerate(keypoints):
            print(f"Person {i + 1}:")
            for j, (x, y, confidence) in enumerate(person_keypoints):
                print(f"  Joint {j + 1}: x={x:.2f}, y={y:.2f}, confidence={confidence:.2f}")
    else:
        print("No keypoints detected in the image.")

# 실행 예제
image_path = "your_image.jpg"  # 분석할 이미지 파일 경로
model_path = "yolov8s-pose.pt"  # YOLOv8 모델 파일 경로
extract_keypoints(image_path, model_path)
