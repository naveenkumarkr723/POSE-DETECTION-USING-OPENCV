import cv2
import mediapipe as mp
import time


class pose_detector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpdraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,
                                     self.detectionCon,self.trackCon)


    def findpose(self,img,draw = True):
        img_bgr = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_bgr)
        if self.results.pose_landmarks:
            if draw:
                self.mpdraw.draw_landmarks(img , self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findposition(self,img,draw = True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),-1)
        return lmlist
def main():
    past_time = 0
    cap = cv2.VideoCapture(0)
    detector = pose_detector()
    while cap.isOpened():
        success, img = cap.read()
        img = detector.findpose(img)
        lmlist = detector.findposition(img,draw=False )
        if len(lmlist)!= 0:
            print(lmlist[15])
            cv2.circle(img, (lmlist[15][1], lmlist[15][2]), 15, (0, 0, 255), -1)

        ctime = time.time()
        fps = 1 / (ctime - past_time)
        past_time = ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        cv2.imshow("pose", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()