# demo.py의 put_text 함수 수정
# putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
# img: 텍스트를 그릴 이미지. NumPy 배열 형식
# text: 이미지에 표시할 텍스트. 문자열 형태.
# org: 텍스트를 그릴 위치. (x, y) 형태의 튜플로, 텍스트의 왼쪽 하단 기준.
# fontFace: 폰트 종류. OpenCV에서 제공하는 폰트 상수 중 하나
# fontScale: 텍스트 크기를 조정하는 배율.
# color: 텍스트 색상. (B, G, R) 형식의 튜플.
# thickness (기본값=1): 텍스트 선의 두께.
# lineType (기본값=cv2.LINE_8): 선의 유형.
# cv2.LINE_4: 4 연결선
# cv2.LINE_8: 8 연결선 (기본)
# cv2.LINE_AA: 안티앨리어싱 선
# bottomLeftOrigin (기본값=False): True로 설정 시, 텍스트의 기준점을 하단에서 위쪽으로 변경.

#노트북 전체 화면의 왼쪽 상단이 (0,0)이며, 오른쪽으로 갈수록 x증가, 아래로 갈수록 y증가

import cv2


def put_text(frame, exercise, count, fps, heart_data, redio):
    global remaining_time
    
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(205 * redio)),
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
    cv2.putText(
        frame, f'Heart: {heart_data}', (int(30 * redio), int(200 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
    