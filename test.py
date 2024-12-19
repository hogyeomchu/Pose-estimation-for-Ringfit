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
        # Read a frame from the video
        success, frame = cap.read()
    
        if success:
            # Set plot size redio for inputs with different resolutions
            frame = cv2.resize(frame, (1280, 800))
            plot_size_redio = max(frame.shape[1] / 1280, frame.shape[0] / 800)

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Preventing errors caused by special scenarios
            if results[0].keypoints.shape[1] == 0:
                if args.show:
                    put_text(frame, 'No Object', counter,
                             score, plot_size_redio)
                    scale = 1280 / max(frame.shape[0], frame.shape[1])
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if args.save_dir is not None:
                    output.write(frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
########################## 고쳐야 할 부분 FSM 넣기
                    
            if fsm.current_state == "ready":
                # 바운딩 박스 넣어보기 안될 수도 있음
                bbox_x, bbox_y, bbox_width, bbox_height = 540, 700, 200, 75
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 0, 255), 2)
                
                # 발 키포인트(15, 16) 확인 및 상태 변경
                if results[0].keypoints is not None:
                    for person_keypoints in results[0].keypoints.data:
                        left_ankle = person_keypoints[15]
                        right_ankle = person_keypoints[16]

                        left_x, left_y, left_conf = left_ankle
                        right_x, right_y, right_conf = right_ankle

                        if (    # bounding box에 발 keypoint가 들어오는지 확인
                            left_conf > 0.5 and bbox_x <= left_x <= bbox_x + bbox_width and bbox_y <= left_y <= bbox_y + bbox_height
                        ) and (
                            right_conf > 0.5 and bbox_x <= right_x <= bbox_x + bbox_width and bbox_y <= right_y <= bbox_y + bbox_height
                        ):
                            start_timer(3)
                            fsm.current_state = "start"
                        else:
                            start_time = None

            if fsm.current_state == "start":            
                # Get hyperparameters
                example_idx = sport_list[args.sport]['example1_idx']
                boundary = sport_list[args.sport]['boundary']

                # Calculate mse
                mse = calculate_mse(results[0].keypoints, example_idx, height, weight)

                # Determine whether to complete once
                if mse < boundary:
                    # 점수 계산
                    #temp_score = calculate_score(results[0].keypoints, mse, boundary)
                    #max_score = max(score, temp_score)
                    
                    if start_time is None:  # 시작 시간 초기화
                        start_time = time.time()
                    elif time.time() - start_time >= 3:  # 3초 이상 경과 확인
                        fsm.current_state = "redo"
                else:
                    start_time = None

            if fsm.current_state == "redo":
                # Get hyperparameters
                example_idx = sport_list[args.sport]['example2_idx']
                boundary = sport_list[args.sport]['boundary']

                # Calculate mse
                mse = calculate_mse(results[0].keypoints, example_idx, height, weight)

                # Determine whether to complete once
                if mse < boundary:
                    # 점수 계산
                    #if args.sport != "squart":
                        #temp_score = calculate_score(results[0].keypoints, mse, boundary)
                        #max_score = max(score, temp_score)

                    if start_time is None:  # 시작 시간 초기화
                        start_time = time.time()
                    elif time.time() - start_time >= 3:  # 1초 이상 경과 확인
                        counter += 1
                        #score = max_score
                        if counter < 10:
                            fsm.current_state = "start"
                        else:
                            fsm.current_state = "finish"
                else:
                    start_time = None

            if fsm.current_state == "finish":
                counter = 0
                args.sport = sports[sport_index]
                sport_index = (sport_index + 1) % 3
                fsm.current_state = "ready"

            print("state: ", fsm.current_state)
############################
            # Visualize the results on the frame
            annotated_frame = plot(
                results[0], plot_size_redio,
                # sport_list[args.sport]['concerned_key_points_idx'],
                # sport_list[args.sport]['concerned_skeletons_idx']
            )
            # annotated_frame = results[0].plot(boxes=False)

            # add relevant information to frame  # fps = round(1000 / results[0].speed['inference'], 2)
            put_text(annotated_frame, args.sport, counter, score, plot_size_redio)
            

            # Display the annotated frame
            if args.show:
                scale = 1280 / max(annotated_frame.shape[0], annotated_frame.shape[1])
                show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow("YOLOv8 Inference", show_frame)

            if args.save_dir is not None:
                output.write(annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
