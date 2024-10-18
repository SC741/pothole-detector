from ultralytics import YOLO
import cv2
#cap = cv2.VideoCapture('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/data/vid_test/istockphoto-937868038-640_adpp_is.mp4')
#cap = cv2.VideoCapture('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/data/vid_test/istockphoto-875261436-640_adpp_is.mp4')
#cap = cv2.VideoCapture('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/data/vid_test/istockphoto-939247874-640_adpp_is.mp4')
cap = cv2.VideoCapture('C:/Users/chara/OneDrive/Desktop/PROGRAMS/yolo/data/vid_test/istockphoto-1319663198-640_adpp_is.mp4')
cap.set(3, 640)
cap.set(4, 480)
model = YOLO('C:\Users\chara\OneDrive\Desktop\PROGRAMS\yolo\runs\detect\train10\weights\last.pt')
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()

        #frame =  cv2.resize(frame,(320,320))
   
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    results1 = model(frame)
    
    
    res_plotted1 = results1[0].plot()
    cv2.imshow("Detected Objects in 2_video", res_plotted1)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
