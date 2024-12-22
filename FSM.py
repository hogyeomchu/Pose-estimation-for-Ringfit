import os
import cv2
import numpy as np
import math
import datetime
import argparse
import pygame
import sys
import time 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy



class State:
    def __init__(self, delay, nstate1=None, nstate2=None, nstate3=None):
        self.delay = delay       # 상태의 지연 시간
        self.nstate1 = nstate1   # 다음 상태 1
        self.nstate2 = nstate2   # 다음 상태 2
        self.nstate3 = nstate3   # 다음 상태 3



# FSM 클래스
class FSM:
    def __init__(self):
        # 상태 초기화
        self.states = {  
            "ready": State(delay=5, nstate1="start", nstate2="start"),
            "start": State(delay=3, nstate1="redo", nstate2="redo"),
            "redo": State(delay=3, nstate1="start", nstate2="finish"),
            "finish": State(delay=0, nstate1="ready", nstate2="ready"),

        }
        self.current_state = "ready"  # 초기 상태 설정

    def transition(self, condition1, condition2, condition3):   #  timer == 0 / timer == 0 & count==10
        # 현재 상태의 객체 가져오기
        state_obj = self.states[self.current_state]

        # 조건에 따라 다음 상태 결정
        if condition1 == 1 and state_obj.nstate1:
            self.current_state = state_obj.nstate1
        if condition2 == 1 and state_obj.nstate2:
            self.current_state = state_obj.nstate2
        else:
            print("유효하지 않은 조건입니다.")

