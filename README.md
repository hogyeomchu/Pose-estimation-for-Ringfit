# Pose-estimation-for-Ringfit

Using jetson nano and ultralytics YOLO model.

# By Daekyo
MSE_function.py = skeleton model 과 input의 x,y좌표를 배열로 인수를 받고, 이를 통해 (x^2+y^2) / n 을 구한다. 파일에서 좌표를 추출하는 건 미포함

put_text.py = 기존 demo.py에 정의된 put_text 함수에 heart beat 아두이노 모듈 값을 불러와 화면에 실시간으로 데이터를 보여주는 부분 추가

heart.py = serial 통신을 이용해서 아두이노 센서로 읽은 값을 가져와서 값을 변수에 저장. + 음악 파일 재생


# By Hokyeom


