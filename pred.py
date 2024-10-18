import cv2
from ultralytics import YOLO
path=cv2.imread('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/data/train/images/242.jpg')
model=YOLO('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/runs/detect/train10/weights/last.pt')
load=model(path)
result=load[0].plot()
cv2.imshow('frame',result)
cv2.waitKey(0)


