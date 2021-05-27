import cv2
import time
import PoseModule as pm

past_time = 0
cap = cv2.VideoCapture(0)
detector = pm.pose_detector()
while cap.isOpened():
    success, img = cap.read()
    img = detector.findpose(img)
    lmlist = detector.findposition(img,draw=False )
    if len(lmlist) != 0:
        print(lmlist[15])
        cv2.circle(img, (lmlist[15][1], lmlist[15][2]),15,(0,0,255), -1)

    ctime = time.time()
    fps = 1 / (ctime - past_time)
    past_time = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    cv2.imshow("pose", img)
    cv2.waitKey(1)
