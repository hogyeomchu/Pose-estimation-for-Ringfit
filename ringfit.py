import os
import cv2
import torch
import numpy as np
import math
import datetime
import argparse
import pygame
import sys
import time
import logging
import threading
import Jetson.GPIO as GPIO
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy

#import timer


sport_list = {
    'lunge': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'boundary': 2000,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]],
        'example1_idx':  [[827.2361450195312, 348.7127685546875, 0.6343599557876587], [0.0, 0.0, 0.07872648537158966], [812.6150512695312, 340.96697998046875, 0.7598568797111511], [0.0, 0.0, 0.07045243680477142], [784.4114990234375, 352.44903564453125, 0.9502374529838562], [768.4188842773438, 423.96343994140625, 0.8846271634101868], [784.532958984375, 414.65484619140625, 0.999079704284668], [0.0, 0.0, 0.21771305799484253], [776.4591674804688, 501.06011962890625, 0.9966193437576294], [0.0, 0.0, 0.15781749784946442], [792.6640625, 562.5013427734375, 0.980634331703186], [748.17041015625, 591.1649780273438, 0.9830169081687927], [781.72412109375, 582.54150390625, 0.9986773133277893], [752.1775512695312, 684.2326049804688, 0.9479348063468933], [917.7450561523438, 629.0517578125, 0.9960450530052185], [634.452392578125, 731.0155029296875, 0.8735394477844238], [900.0355834960938, 739.8284301757812, 0.9781225919723511]], # 여기에 예시 스켈레톤 넣기
        'example2_idx': [[791.9453125, 363.17181396484375, 0.8146561980247498], [0.0, 0.0, 0.18356390297412872], [788.5883178710938, 360.59906005859375, 0.9212821125984192], [0.0, 0.0, 0.06203003227710724], [760.001220703125, 366.2190246582031, 0.9566388130187988], [725.8297119140625, 410.266357421875, 0.8075394630432129], [744.4752197265625, 431.987060546875, 0.9988507032394409], [0.0, 0.0, 0.22919785976409912], [724.746826171875, 504.1793212890625, 0.9977599382400513], [0.0, 0.0, 0.1730046570301056], [750.7670288085938, 562.2422485351562, 0.9869770407676697], [768.258056640625, 579.3230590820312, 0.9870125651359558], [761.249755859375, 607.5520629882812, 0.9991191029548645], [881.926513671875, 611.0643310546875, 0.9667600393295288], [776.908203125, 722.4578857421875, 0.9976578950881958], [857.5050659179688, 734.7716064453125, 0.9517019987106323], [614.5533447265625, 739.4373168945312, 0.99078768491745]]
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'boundary': 1,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]],
        'example1_idx': [],
        'example2_idx': []
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'boundary': 10000,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [17, 15], [15, 13]],
        'example1_idx': [[790.8934936523438, 370.8782958984375, 0.9554082751274109], [0.0, 0.0, 0.3667287528514862], [780.343017578125, 355.3907470703125, 0.9803277850151062], [0.0, 0.0, 0.03333147615194321], [736.9127197265625, 359.50872802734375, 0.9780341982841492], [679.8504638671875, 417.9067687988281, 0.9015007019042969], [692.560791015625, 423.825927734375, 0.9957813024520874], [742.5250244140625, 501.7216796875, 0.5579652190208435], [787.887939453125, 532.1363525390625, 0.9937047362327576], [799.5919189453125, 455.236083984375, 0.6515821218490601], [813.8155517578125, 444.1479797363281, 0.9879148006439209], [523.1993408203125, 546.7066040039062, 0.9530515074729919], [520.2064819335938, 554.30029296875, 0.9888673424720764], [710.7117919921875, 564.3704833984375, 0.9626505374908447], [698.611083984375, 583.9825439453125, 0.9933033585548401], [644.0439453125, 708.7687377929688, 0.9193461537361145], [603.1053466796875, 730.88427734375, 0.9693002104759216]],
        'example2_idx': [[726.5889892578125, 90.649658203125, 0.9245054125785828], [0.0, 0.0, 0.33452704548835754], [719.0135498046875, 75.04046630859375, 0.9768596291542053], [0.0, 0.0, 0.04215420037508011], [673.3643798828125, 71.355224609375, 0.9657612442970276], [634.0013427734375, 158.72515869140625, 0.5154595971107483], [655.8916625976562, 161.07513427734375, 0.9977946281433105], [0.0, 0.0, 0.05190538614988327], [652.5096435546875, 298.1383056640625, 0.9959108829498291], [0.0, 0.0, 0.0905299261212349], [697.6686401367188, 401.5651550292969, 0.986933708190918], [645.534423828125, 396.4898681640625, 0.9457389712333679], [661.8905029296875, 401.6611328125, 0.9973733425140381], [641.5545654296875, 554.7891845703125, 0.9502384662628174], [669.8883056640625, 566.0969848632812, 0.9975005984306335], [624.8091430664062, 712.2351684570312, 0.9154536128044128], [627.0443115234375, 724.532470703125, 0.9882832765579224]]

    }
}


logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


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
        print("No keypoints detected.")
        return 99999

    # Automatically detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move key_points data to the device
    key_points = key_points.data.to(device)

    # Convert example to tensor and move to the device
    example_np = np.array(example)  # example의 shape은 (17, 3) 형태로 가정
    example_tensor = torch.tensor(example_np, device=device)

    # Filter keypoints based on confidence
    valid_indices = (key_points[0, :, 2] > confidence_threshold) & (example_tensor[:, 2] > confidence_threshold)

    # Ensure valid_indices is boolean tensor
    valid_indices = valid_indices.bool()

    # Filter valid keypoints and example points
    valid_key_coords = key_points[0, valid_indices, :2]
    valid_example_coords = example_tensor[valid_indices, :2]

    # Ensure valid keypoints to compare
    if valid_key_coords.size(0) == 0:
        print("No keypoints meet the confidence threshold.")
        return 99999

    # Calculate MSE using specific formula
    x_diff = ((valid_key_coords[:, 0] - key_points[0, 16, 0]) - (valid_example_coords[:, 0] - example_tensor[16, 0]))
    y_diff = ((valid_key_coords[:, 1] - key_points[0, 16, 1]) / (height * 0.01) - (valid_example_coords[:, 1] - example_tensor[16, 1]) / 1.84)

    mse = torch.mean(x_diff ** 2 + y_diff ** 2).item()

    print("Calculated MSE:", mse)
    return mse

def calculate_score(key_points, mse, boundary, confidence_threshold=0.5):
    if key_points is None or len(key_points) == 0:
        print("No keypoints detected.")
        return 0  # 점수를 0으로 반환

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    key_points = key_points.data.to(device)  # Keypoints를 텐서로 변환 후 디바이스로 이동
    key_points = key_points.squeeze(0)  # 불필요한 차원 축소

    # 기본 점수
    basic_score = 50

    # 키포인트 인덱스
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    # 1. Hip Score (엉덩이 vs 무릎의 y 좌표)
    hip_diff = 0
    hip_score = 0
    if key_points[LEFT_HIP, 2] > confidence_threshold and key_points[LEFT_KNEE, 2] > confidence_threshold:
        left_hip_y = key_points[LEFT_HIP, 1]
        left_knee_y = key_points[LEFT_KNEE, 1]
        hip_diff += (left_knee_y - left_hip_y)
    if key_points[RIGHT_HIP, 2] > confidence_threshold and key_points[RIGHT_KNEE, 2] > confidence_threshold:
        right_hip_y = key_points[RIGHT_HIP, 1]
        right_knee_y = key_points[RIGHT_KNEE, 1]
        hip_diff += (right_knee_y - right_hip_y)

    if hip_diff > 0:
        hip_score = max(0, min(20, (hip_diff / 50) * 20))  # 정규화하여 최대 20점

    # 2. Upper Score (어깨와 엉덩이/무릎 중앙 비교)
    upper_score = 0
    shoulder_center_x = None
    hip_knee_center_x = None

    if key_points[LEFT_SHOULDER, 2] > confidence_threshold and key_points[RIGHT_SHOULDER, 2] > confidence_threshold:
        left_shoulder_x = key_points[LEFT_SHOULDER, 0]
        right_shoulder_x = key_points[RIGHT_SHOULDER, 0]
        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2

    if (key_points[LEFT_HIP, 2] > confidence_threshold and key_points[LEFT_KNEE, 2] > confidence_threshold and
        key_points[RIGHT_HIP, 2] > confidence_threshold and key_points[RIGHT_KNEE, 2] > confidence_threshold):
        left_hip_knee_x = (key_points[LEFT_HIP, 0] + key_points[LEFT_KNEE, 0]) / 2
        right_hip_knee_x = (key_points[RIGHT_HIP, 0] + key_points[RIGHT_KNEE, 0]) / 2
        hip_knee_center_x = (left_hip_knee_x + right_hip_knee_x) / 2

    if shoulder_center_x is not None and hip_knee_center_x is not None:
        upper_diff = abs(shoulder_center_x - hip_knee_center_x)
        upper_score = max(0, min(15, (1 - (upper_diff / 100)) * 15))  # 정규화하여 최대 15점

    # 3. Knee Score (무릎과 발의 x 좌표 비교)
    knee_score = 0
    knee_diff = 0
    if key_points[LEFT_KNEE, 2] > confidence_threshold and key_points[LEFT_ANKLE, 2] > confidence_threshold:
        left_knee_x = key_points[LEFT_KNEE, 0]
        left_ankle_x = key_points[LEFT_ANKLE, 0]
        knee_diff += abs(left_knee_x - left_ankle_x)
    if key_points[RIGHT_KNEE, 2] > confidence_threshold and key_points[RIGHT_ANKLE, 2] > confidence_threshold:
        right_knee_x = key_points[RIGHT_KNEE, 0]
        right_ankle_x = key_points[RIGHT_ANKLE, 0]
        knee_diff += abs(right_knee_x - right_ankle_x)

    if knee_diff > 0:
        knee_score = max(0, min(15, (1 - (knee_diff / 50)) * 15))  # 정규화하여 최대 15점

    # 최종 점수 계산
    score = basic_score + hip_score + upper_score + knee_score

    return int(score)

#######################################
interrupt_pin = 16
timer_state = {
    "running": False,
    "over": False,
}
timer_event = threading.Event()
lock = threading.Lock()  # 동기화를 위한 Lock 객체

# GPIO 초기화
def setup_gpio():
    GPIO.setmode(GPIO.BOARD)  # GPIO 핀 번호 설정 방식
    GPIO.setup(interrupt_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # 핀 설정 (입력 핀으로 설정)

# 타이머 종료 함수
def stop_timer():
    with lock:
        timer_event.set()  # 이벤트를 설정하여 타이머를 중단
        timer_state["running"] = False
        print("타이머 중단 및 초기화")

# 타이머 시작 함수
def start_timer(duration):
    with lock:
        if timer_state["running"]:  # 이미 타이머가 실행 중이면 중단
            stop_timer()

        timer_state["running"] = True
        timer_state["over"] = False
        timer_event.clear()  # 이벤트 초기화
        print("타이머 시작!")

    def timer_task():
        nonlocal duration
        while duration > 0:
            if timer_event.is_set():  # 이벤트가 설정되면 중단
                return
            print(f"남은 시간: {duration}초")
            time.sleep(1)
            duration -= 1

        with lock:
            timer_state["running"] = False
            timer_state["over"] = True
            print("타이머 종료!")

    threading.Thread(target=timer_task, daemon=True).start()

# 상태 확인 함수
def is_timer_running():
    with lock:
        return timer_state["running"]

def is_timer_over():
    with lock:
        return timer_state["over"]


# GPIO 정리 함수
def cleanup_gpio():
    GPIO.cleanup()
    print("GPIO 정리 완료")
###########################################


def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
    class _Annotator(Annotator):

        def kpts(self, kpts, shape=(1280, 1280), radius=5, line_thickness=2, kpt_line=True):
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


def put_text(frame, exercise, count, score, redio):
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
    #cv2.putText(
    #    frame, f'FPS: {fps}', (int(30 * redio), int(150 * redio)), 0, 0.9 * redio,
    #    (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    cv2.putText(
        frame, f'Score: {score}', (int(30 * redio), int(150 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s-pose.pt', type=str, help='path to model weight')
    parser.add_argument('--sport', default='squat', type=str,
                        help='Currently supported "lunge", "pushup" and "squat"')
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
    model = YOLO(args.model, verbose=False)

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
    state = "ready"  # ready, start, redo, finish
    sports = list(sport_list.keys())
    sport_index = 0

    reaching = False
    reaching_last = False
    state_keep = False
    start_time = None 
    counter = 0
    max_score = 0
    score = 0
    time_ck = 0
    height, weight = 180, 70
    # height, weight = get_height_and_weight()
    setup_gpio()
    # Loop through the video frames
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
                    
            if state == "ready":
                # 바운딩 박스 넣어보기 안될 수도 있음
                bbox_x, bbox_y, bbox_width, bbox_height = 490, 650, 300, 100
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 0, 255), 2)
                
                # 발 키포인트(15, 16) 확인 및 상태 변경
                if results[0].keypoints is not None:
                    for person_keypoints in results[0].keypoints.data:
                        left_ankle = person_keypoints[15]
                        right_ankle = person_keypoints[16]

                        left_x, left_y, left_conf = left_ankle
                        right_x, right_y, right_conf = right_ankle

                        if (
                            (left_conf > 0.5 and bbox_x <= left_x <= bbox_x + bbox_width and bbox_y <= left_y <= bbox_y + bbox_height)
                            or (right_conf > 0.5 and bbox_x <= right_x <= bbox_x + bbox_width and bbox_y <= right_y <= bbox_y + bbox_height)
                        ):
                            if time_ck == 0:
                                start_timer(3)
                                time_ck == 1
                            
                            if is_timer_over() and time_ck == 1:
                                state = "start"
                                time_ck = 0

                        else:
                            if is_timer_running():
                                stop_timer()
                                time_ck == 0

            if state == "start":            
                # Get hyperparameters
                example_idx = sport_list[args.sport]['example1_idx']
                boundary = sport_list[args.sport]['boundary']

                # Calculate mse
                mse = calculate_mse(results[0].keypoints, example_idx, height, weight)

                # Determine whether to complete once
                if mse < boundary:
                    # 점수 계산
                    temp_score = calculate_score(results[0].keypoints, mse, boundary)
                    max_score = max(max_score, temp_score)
                    
                    if start_time is None:  # 시작 시간 초기화
                        start_time = time.time()
                    elif time.time() - start_time >= 3:  # 3초 이상 경과 확인
                        state = "redo"
                else:
                    start_time = None

            if state == "redo":
                # Get hyperparameters
                example_idx = sport_list[args.sport]['example2_idx']
                boundary = sport_list[args.sport]['boundary']

                # Calculate mse
                mse = calculate_mse(results[0].keypoints, example_idx, height, weight)

                # Determine whether to complete once
                if mse < boundary:
                    # 점수 계산
                    if args.sport != "squart":
                        temp_score = calculate_score(results[0].keypoints, mse, boundary)
                        max_score = max(max_score, temp_score)

                    if start_time is None:  # 시작 시간 초기화
                        start_time = time.time()
                    elif time.time() - start_time >= 3:  # 1초 이상 경과 확인
                        counter += 1
                        score = max_score
                        max_score = 0
                        if counter < 10:
                            state = "start"
                        else:
                            state = "finish"
                else:
                    start_time = None

            if state == "finish":
                counter = 0
                args.sport = sports[sport_index]
                sport_index = (sport_index + 1) % 3
                state = "ready"

            #print("state: ", state)
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

    # Release the video capture object and close the display window
    cap.release()
    GPIO.cleanup()
    if args.save_dir is not None:
        output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
