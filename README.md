# Pose-estimation-for-Ringfit

Using jetson nano and ultralytics YOLO model.


MSE_function.py = skeleton model 과 input의 x,y좌표를 배열로 인수를 받고, 이를 통해 (x^2+y^2) / n 을 구한다. 파일에서 좌표를 추출하는 건 미포함