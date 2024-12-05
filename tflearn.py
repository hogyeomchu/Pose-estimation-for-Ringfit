from ultralytics import YOLO

# 1. 사전 학습된 모델 불러오기
model = YOLO("yolov8s-pose.pt")  # 기존 사전 학습된 모델

# 2. 데이터셋 경로 설정 (YOLO 포맷)
data = {
    'train': 'path/to/dataset/images/train',  # 학습 이미지 경로
    'val': 'path/to/dataset/images/val',      # 검증 이미지 경로
    'names': {0: 'person'}                    # 클래스 이름
}

# 3. 하이퍼파라미터 설정
model.train(
    data=data,                  # 데이터셋 경로
    epochs=50,                  # 학습 에포크 수
    batch=16,                   # 배치 크기
    imgsz=640,                  # 입력 이미지 크기
    device=0,                   # GPU 사용 (0번 GPU)
    workers=4,                  # 데이터 로더 워커 수
    name='yolov8s_pose_finetune',  # 프로젝트 이름
    pretrained=True             # 사전 학습된 가중치 사용
)
