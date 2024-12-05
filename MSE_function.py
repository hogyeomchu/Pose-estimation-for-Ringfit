import numpy as np

# x1 y1 = skeleton model에서 keypoint의 x, y 좌표
# x2 y2 = 우리 input의 x좌표 y좌표

length = 17 # total number of keypoints 

# a =  keypoints[0][i][0] b = keypoints[0][i][1]

def mse(a,b, c, d):       # x1 y1 x2 y2 order : 각각은 17개 원소 배열 
    for i in range(length):
        sum += (a[i] - c[i])**2

    for j in range(length):
        sum += (b[i] - d[i])**2
    
    sum = sum / 17
    
    return sum 















