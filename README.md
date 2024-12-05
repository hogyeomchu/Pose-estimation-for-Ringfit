# Pose-estimation-for-Ringfit

Using jetson nano and ultralytics YOLO model.

# By Daekyo
MSE_function.py = skeleton model 과 input의 x,y좌표를 배열로 인수를 받고, 이를 통해 (x^2+y^2) / n 을 구한다. 파일에서 좌표를 추출하는 건 미포함

put_text.py = 기존 demo.py에 정의된 put_text 함수에 heart beat 아두이노 모듈 값을 불러와 화면에 실시간으로 데이터를 보여주는 부분 추가

heart.py = serial 통신을 이용해서 아두이노 센서로 읽은 값을 가져와서 값을 변수에 저장. + 음악 파일 재생


# By Hogyeom
camera.py = 카메라로 스켈레톤 따오기

demo.py = 그냥 긁어온거

ringfit.py = demo.py에서 수정하고 있는 것 아마 메인코드가 될 것 같음

tflearn = 전이학습 그냥 대충 구현해봄 쓸일 없을듯

skeleton = 고정된 좌표 줬을 때 스켈레톤 띄워지는지 테스트용

keypoint.py = ㅇㅣㅁㅣㅈㅣ keypoint ㅂㅕㄴㅎㅗㅏㄴ
