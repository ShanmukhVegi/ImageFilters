import cv2
import numpy as np
import dlib
import math
from scipy.spatial import distance as dist

class Mustache_Filter:
    def __init__(self):
        self.landmark_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/Users/shanmukhvegi/Desktop/Mini-Project/shape_predictor_68_face_landmarks.dat")
    
    def start_process(self):
        mustache_image = cv2.imread("/Users/shanmukhvegi/Desktop/Mini-Project/Mustache_Filter/mustache.png")
        vid_capture=cv2.VideoCapture(0)
        vid, frame = vid_capture.read()
        rows, cols, _vid= frame.shape
        mustache_mask = np.zeros((rows, cols))

        while True:
            vid, frame = vid_capture.read()
            mustache_mask.fill(0)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.landmark_detector(frame)
            for face in faces:
                landmarks = self.predictor(gray_frame, face)
                mouthleft= (landmarks.part(48).x, landmarks.part(48).y)
                mouthright= (landmarks.part(54).x, landmarks.part(54).y)
                nosebottom = (landmarks.part(33).x, landmarks.part(33).y)
                mouthmiddle = (landmarks.part(51).x, landmarks.part(51).y)
                mustache_width = 2*int(dist.euclidean(mouthleft,mouthright))
                mustache_height = 5*int(dist.euclidean(nosebottom,mouthmiddle))

                up_center = (int(nosebottom[0]-mustache_width/2),int(nosebottom[1]-mustache_height/2))
                down_center = (int(mouthmiddle[0]),int(mouthmiddle[1]-mustache_height))
                mustache_area = frame[up_center[1]: up_center[1] + mustache_height,up_center[0]: up_center[0] + mustache_width]
                mustache_img = cv2.resize(mustache_image, (mustache_width, mustache_height),interpolation = cv2.INTER_AREA)
                mustache_img_gray = cv2.cvtColor(mustache_img, cv2.COLOR_BGR2GRAY)
                try:
                    final_frame = cv2.add(mustache_area, mustache_img)
                except:
                    continue
                
                frame[up_center[1]: up_center[1] + mustache_height,up_center[0]: up_center[0] + mustache_width] = final_frame
            cv2.imshow("final Mustache", frame)
            if(cv2.waitKey(1)==27):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)