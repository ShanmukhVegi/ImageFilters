import numpy as np  
import cv2  
import dlib  
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull  

class Eye_Filter:
    def __init__(self):
        self.PREDICTOR_PATH = "../Mini-Project/shape_predictor_68_face_landmarks.dat"  
        self.RIGHT_EYE_POINTS = list(range(36, 42))  
        self.LEFT_EYE_POINTS = list(range(42, 48)) 
        self.detector = dlib.get_frontal_face_detector()  
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        
    def eye_size(self,eye):
        eyeWidth = dist.euclidean(eye[0], eye[3])  
        hull = ConvexHull(eye)  
        eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
        eyeCenter = eyeCenter.astype(int)
        return int(eyeWidth), eyeCenter

    def place_eye(self, eyeCenter, eyeSize):
        eyeSize = int(eyeSize * 1.5)
        x1 = int(eyeCenter[0,0] - (eyeSize/2))  
        x2 = int(eyeCenter[0,0] + (eyeSize/2))  
        y1 = int(eyeCenter[0,1] - (eyeSize/2))  
        y2 = int(eyeCenter[0,1] + (eyeSize/2))  
           
        h, w = self.frame.shape[:2]
        if x1 < 0:
            x1 = 0  
        if y1 < 0:  
            y1 = 0  
        if x2 > w:  
            x2 = w  
        if y2 > h:  
            y2 = h 
        eyeOverlayWidth = x2 - x1  
        eyeOverlayHeight = y2 - y1  
        eyeOverlay = cv2.resize(self.imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
        mask = cv2.resize(self.orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
        mask_inv = cv2.resize(self.orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)   
        roi = self.frame[y1:y2, x1:x2] 
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv) 
        roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)
        dst = cv2.add(roi_bg,roi_fg) 
        self.frame[y1:y2, x1:x2] = dst

    def start_process(self):
        self.imgEye = cv2.imread('../Mini-Project/Eye_Filter/Eye.png',-1) 
        self.orig_mask = self.imgEye[:,:,3]  
        self.orig_mask_inv = cv2.bitwise_not(self.orig_mask)   
        self.imgEye = self.imgEye[:,:,0:3]  
        origEyeHeight, origEyeWidth = self.imgEye.shape[:2]
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, self.frame = video_capture.read()  
            if ret:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  
           
                rects = self.detector(gray, 0)  
           
                for rect in rects:  
                  x = rect.left()  
                  y = rect.top()  
                  x1 = rect.right()  
                  y1 = rect.bottom()  
           
                  landmarks = np.matrix([[p.x, p.y] for p in self.predictor(self.frame, rect).parts()])  
           
                  left_eye = landmarks[self.LEFT_EYE_POINTS]  
                  right_eye = landmarks[self.RIGHT_EYE_POINTS] 
           
                  leftEyeSize, leftEyeCenter = self.eye_size(left_eye)  
                  rightEyeSize, rightEyeCenter = self.eye_size(right_eye)  
           
                  self.place_eye( leftEyeCenter, leftEyeSize)  
                  self.place_eye( rightEyeCenter, rightEyeSize)  
           
                cv2.imshow("Eye_Filter", self.frame)
                if(cv2.waitKey(1)==27):
                    cv2.destroyWindow("Eye_Filter")
                    break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)