import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# TensorRT 모델 로드 함수
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# TensorRT 추론 실행 함수
def infer(engine, input_data):
    with engine.create_execution_context() as context:
        # 입력 및 출력 바인딩 인덱스 확인
        input_binding_idx = engine.get_binding_index("images")
        output_binding_idx = engine.get_binding_index("output_0")

        # 입력 및 출력 크기 계산
        input_shape = engine.get_binding_shape(input_binding_idx)
        output_shape = engine.get_binding_shape(output_binding_idx)

        # 호스트 및 디바이스 메모리 할당
        h_input = np.ascontiguousarray(input_data, dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(trt.volume(output_shape) * h_input.itemsize)

        # 메모리에 데이터 복사 및 추론 실행
        cuda.memcpy_htod(d_input, h_input)
        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings)

        # 출력 데이터 가져오기
        h_output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_output, d_output)
        return h_output

# 데이터 전처리 함수 (이미지 전처리)
def preprocess_image(image, input_size=(640, 640)):
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# 데이터 후처리 함수 (키포인트 추출)
def postprocess_output(output, threshold=0.5):
    keypoints = output.reshape(-1, 3)
    valid_keypoints = keypoints[keypoints[:, 2] > threshold]
    return valid_keypoints

# TensorRT 기반 YOLO 추론 메인 함수
def main():
    engine_path = "yolov8s-pose.engine"  # 변환된 TensorRT 엔진 파일 경로
    engine = load_engine(engine_path)

    # 입력 소스 설정 (웹캠 또는 비디오 파일)
    cap = cv2.VideoCapture(0)  # 0: 기본 웹캠

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 전처리
        input_data = preprocess_image(frame)

        # TensorRT 추론
        output = infer(engine, input_data)

        # 후처리
        keypoints = postprocess_output(output)

        # 시각화 (키포인트 그리기)
        for keypoint in keypoints:
            x, y, conf = keypoint
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # 결과 표시
        cv2.imshow("YOLOv8 Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
