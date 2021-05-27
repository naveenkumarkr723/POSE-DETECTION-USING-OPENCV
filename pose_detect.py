import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpdraw = mp.solutions.drawing_utils
past_time =0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    img_bgr = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = pose.process(img_bgr)
    #print(results.pose_landmarks)

    if results.pose_landmarks:
        mpdraw.draw_landmarks(img , results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx,cy = int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),-1)


    ctime = time.time()
    fps = 1/(ctime-past_time)
    past_time = ctime

    cv2.putText(img , str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)
    cv2.imshow("pose", img)
    cv2.waitKey(1)