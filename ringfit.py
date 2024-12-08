import os
import cv2
import numpy as np
import math
import datetime
import argparse
import pygame
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy



sport_list = {
    'sit-up': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]],
        'example_idx' : [1] # 여기에 예시 스켈레톤 넣기
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]],
        'example_idx' : [1]
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [17, 15], [15, 13]],
        'example_idx' : [1]
    }
}

attention_idx = [1] # 차렷자세 예시 스켈레톤

def calculate_angle(key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        # Calculate the slope of two straight lines
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # Convert radians to angles
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
    return angle

################# MSE 계산

def calculate_mse(key_points, example, height, weight, confidence_threshold=0.5):
    # Ensure key_points is not empty
    if key_points is None or len(key_points) == 0:
        raise ValueError("No keypoints detected.")

    # Convert key_points to numpy array and normalize
    key_points_np = key_points[0].cpu().numpy()  # Assume single person; Shape: (17, 3)
    key_coords = key_points_np[:, :2]  # Extract (x, y) coordinates
    confidences = key_points_np[:, 2]  # Extract confidence values

    # Normalize keypoint coordinates by image dimensions
    key_coords[:, 0] /= weight
    key_coords[:, 1] /= height

    # Normalize example coordinates
    example_np = np.array(example)
    example_normalized = example_np / np.array([weight, height])

    # Filter keypoints based on confidence
    valid_indices = confidences > confidence_threshold
    valid_key_coords = key_coords[valid_indices]
    valid_example_coords = example_normalized[valid_indices]

    # Ensure there are valid keypoints to compare
    if len(valid_key_coords) == 0:
        raise ValueError("No keypoints meet the confidence threshold.")

    # Calculate MSE
    mse = np.mean((valid_key_coords - valid_example_coords) ** 2)

    return mse

#################

def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
    class _Annotator(Annotator):

        def kpts(self, kpts, shape=(640, 640), radius=5, line_thickness=2, kpt_line=True):
            """Plot keypoints on the image.

            Args:
                kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
                shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
                radius (int, optional): Radius of the drawn keypoints. Default is 5.
                kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                           for human pose. Default is True.
                line_thickness (int, optional): thickness of the kpt_line. Default is 2.

            Note: `kpt_line=True` currently only supports human pose plotting.
            """
            if self.pil:
                # Convert to numpy first
                self.im = np.asarray(self.im).copy()
            nkpt, ndim = kpts.shape
            is_pose = nkpt == 17 and ndim == 3
            kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
            colors = Colors()
            for i, k in enumerate(kpts):
                if show_points is not None:
                    if i not in show_points:
                        continue
                color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)),
                               int(radius * plot_size_redio), color_k, -1, lineType=cv2.LINE_AA)

            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    if show_skeleton is not None:
                        if sk not in show_skeleton:
                            continue
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             thickness=int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))
    if pose_result.keypoints is not None:
        for k in reversed(pose_result.keypoints.data):
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)
    return annotator.result()


def put_text(frame, exercise, count, fps, redio):
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(163 * redio)),
        (55, 104, 0), -1
    )

    if exercise in sport_list.keys():
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(100 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio), int(150 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s-pose.pt', type=str, help='path to model weight')
    parser.add_argument('--sport', default='squat', type=str,
                        help='Currently supported "sit-up", "pushup" and "squat"')
    parser.add_argument('--input', default="0", type=str, help='path to input video')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save output')
    parser.add_argument('--show', default=True, type=bool, help='show the result')
    args = parser.parse_args()
    return args


def get_height_and_weight():
    # Pygame 초기화
    pygame.init()

    # 화면 크기 설정 (예: 800x600)
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("키와 몸무게 입력")

    # 폰트 설정
    font = pygame.font.Font(None, 48)  # 더 큰 화면에 맞춰 글씨 크기를 조정

    # 색상 설정
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)

    # 입력 필드 초기화
    height_text = ""
    weight_text = ""
    active_field = None

    # 결과 저장
    height = None
    weight = None

    # 배경 이미지 로드
    try:
        background_image = pygame.image.load("background.jpg")  # 여기에 원하는 이미지 파일 경로
        background_image = pygame.transform.scale(background_image, (screen_width, screen_height))  # 화면 크기에 맞게 조정
    except FileNotFoundError:
        print("배경 이미지 파일을 찾을 수 없습니다. 기본 하얀 배경을 사용합니다.")
        background_image = None

    # 루프
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 마우스 클릭으로 활성 필드 변경
                if 300 < event.pos[0] < 600 and 200 < event.pos[1] < 260:
                    active_field = "height"
                elif 300 < event.pos[0] < 600 and 300 < event.pos[1] < 360:
                    active_field = "weight"
                else:
                    active_field = None
            elif event.type == pygame.KEYDOWN:
                if active_field == "height":
                    if event.key == pygame.K_RETURN:  # 엔터키로 입력 완료
                        height = height_text
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:  # 백스페이스로 삭제
                        height_text = height_text[:-1]
                    else:
                        height_text += event.unicode
                elif active_field == "weight":
                    if event.key == pygame.K_RETURN:
                        weight = weight_text
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:
                        weight_text = weight_text[:-1]
                    else:
                        weight_text += event.unicode

        # 키와 몸무게가 입력되었는지 확인
        if height and weight:
            running = False

        # 배경 이미지 그리기
        if background_image:
            screen.blit(background_image, (0, 0))
        else:
            screen.fill(WHITE)  # 기본 하얀 배경

        # 텍스트 렌더링
        height_label = font.render("Height:", True, BLACK)
        weight_label = font.render("Weight:", True, BLACK)

        # 입력 박스 그리기 (흰색 채우기)
        pygame.draw.rect(screen, WHITE, (300, 200, 300, 60))  # 흰색으로 채움
        pygame.draw.rect(screen, WHITE, (300, 300, 300, 60))  # 흰색으로 채움
        pygame.draw.rect(screen, GRAY if active_field == "height" else BLACK, (300, 200, 300, 60), 2)  # 테두리
        pygame.draw.rect(screen, GRAY if active_field == "weight" else BLACK, (300, 300, 300, 60), 2)  # 테두리

        # 입력 텍스트 렌더링
        height_surface = font.render(height_text, True, BLACK)
        weight_surface = font.render(weight_text, True, BLACK)

        # 화면에 텍스트와 입력칸 표시
        screen.blit(height_label, (150, 215))  # 높이 텍스트 위치 조정
        screen.blit(weight_label, (150, 315))  # 몸무게 텍스트 위치 조정
        screen.blit(height_surface, (310, 215))  # 입력 필드 텍스트 위치 조정
        screen.blit(weight_surface, (310, 315))  # 입력 필드 텍스트 위치 조정

        # 화면 업데이트
        pygame.display.flip()

    # Pygame 종료
    pygame.quit()

    # 키와 몸무게 반환
    print("키 :", height)
    print("몸무게 :", weight)
    return float(height), float(weight)


def main():
    # Obtain relevant parameters
    args = parse_args()
    # Load the YOLOv8 model
    model = YOLO(args.model)

    # Open the video file or camera
    if args.input.isnumeric():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    # For save result video
    if args.save_dir is not None:
        save_dir = os.path.join(
            args.save_dir, args.sport,
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    # Set variables to record motion status
    state = "ready"  # ready, start, reset, finish

    reaching = False
    reaching_last = False
    state_keep = False
    counter = 0
    #height, weight = input("키와 몸무게를 입력하세요: ").split()
    #print("키: ", height)
    #print("몸무게: ", weight)
    height, weight = get_height_and_weight()


    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
            
        if success:
            # Set plot size redio for inputs with different resolutions
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Preventing errors caused by special scenarios
            if results[0].keypoints.shape[1] == 0:
                if args.show:
                    put_text(frame, 'No Object', counter,
                             round(1000 / results[0].speed['inference'], 2), plot_size_redio)
                    scale = 640 / max(frame.shape[0], frame.shape[1])
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if args.save_dir is not None:
                    output.write(frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
########################## 고쳐야 할 부분 FSM 넣기
                    
            if state == "ready":
                # 바운딩 박스 넣어보기 안될 수도 있음
                bbox_x, bbox_y, bbox_width, bbox_height = 170, 400, 300, 50
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 2)
                
                # 발 키포인트(15, 16) 확인 및 상태 변경
                if results[0].keypoints is not None:
                    for person_keypoints in results[0].keypoints.data:
                        left_ankle = person_keypoints[15]
                        right_ankle = person_keypoints[16]

                        left_x, left_y, left_conf = left_ankle
                        right_x, right_y, right_conf = right_ankle

                        if (
                            left_conf > 0.5 and bbox_x <= left_x <= bbox_x + bbox_width and bbox_y <= left_y <= bbox_y + bbox_height
                        ) or (
                            right_conf > 0.5 and bbox_x <= right_x <= bbox_x + bbox_width and bbox_y <= right_y <= bbox_y + bbox_height
                        ):
                            state = "start"
                            break

            if state == "start":            
                # Get hyperparameters
                left_points_idx = sport_list[args.sport]['left_points_idx']
                right_points_idx = sport_list[args.sport]['right_points_idx']

                example_idx = sport_list[args.sport]['example_idx']

                # Calculate angle
                angle = calculate_angle(results[0].keypoints, left_points_idx, right_points_idx)

                error = calculate_mse(results[0].keypoints, example_idx, height, weight)

                # Determine whether to complete once
                if angle < sport_list[args.sport]['maintaining']:
                    reaching = True
                if angle > sport_list[args.sport]['relaxing']:
                    reaching = False

                if reaching != reaching_last:
                    reaching_last = reaching
                    if reaching:
                        state_keep = True
                    if not reaching and state_keep:
                        counter += 1
                        state_keep = False


############################
            # Visualize the results on the frame
            annotated_frame = plot(
                results[0], plot_size_redio,
                # sport_list[args.sport]['concerned_key_points_idx'],
                # sport_list[args.sport]['concerned_skeletons_idx']
            )
            # annotated_frame = results[0].plot(boxes=False)

            # add relevant information to frame
            put_text(
                annotated_frame, args.sport, counter, round(1000 / results[0].speed['inference'], 2), plot_size_redio)
            

            # Display the annotated frame
            if args.show:
                scale = 640 / max(annotated_frame.shape[0], annotated_frame.shape[1])
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

    # Release the video capture object and close the display window
    cap.release()
    if args.save_dir is not None:
        output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
