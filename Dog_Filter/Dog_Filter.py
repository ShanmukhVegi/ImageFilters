import cv2
import numpy as np
import dlib
import math
from scipy.spatial import distance as dist
import time

class Dog_Filter:
    def __init__(self):
        self.landmark_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/Users/shanmukhvegi/Desktop/Mini-Project/shape_predictor_68_face_landmarks.dat")
        
    def start_process(self):
        dog_image = cv2.imread("/Users/shanmukhvegi/Desktop/Mini-Project/Dog_Filter/doggy.png")
        vid_capture=cv2.VideoCapture(0)
        vid, frame = vid_capture.read()
        rows, cols, _vid= frame.shape
        dog_mask = np.zeros((rows, cols))

        while True:
            vid, frame = vid_capture.read()
            dog_mask.fill(0)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.landmark_detector(frame)
            for face in faces:
                landmarks = self.predictor(gray_frame, face)
                nosetop = (landmarks.part(29).x, landmarks.part(29).y)
                nosemid = (landmarks.part(30).x, landmarks.part(30).y)
                noseleft = (landmarks.part(31).x, landmarks.part(31).y)
                noseright = (landmarks.part(35).x, landmarks.part(35).y)
                nosebottom = (landmarks.part(33).x, landmarks.part(33).y)
                dog_width = 3*int(dist.euclidean(noseleft,noseright))
                dog_height = 8*int(dist.euclidean(nosebottom,nosemid))
                up_center = (int(nosemid[0] - dog_width / 2),int(nosemid[1] - dog_height / 2))
                down_center = (int(nosemid[0] + dog_width / 2),int(nosemid[1] + dog_height / 2))
                dog_area = frame[up_center[1]: up_center[1] + dog_height,up_center[0]: up_center[0] + dog_width]
                dog_img = cv2.resize(dog_image, (dog_width, dog_height))
                dog_img_gray = cv2.cvtColor(dog_img, cv2.COLOR_BGR2GRAY)
                try:final_frame = cv2.add(dog_area, dog_img)
                except:continue
                frame[up_center[1]: up_center[1] + dog_height,up_center[0]: up_center[0] + dog_width] = final_frame
            cv2.imshow("final dog", frame)
            if(cv2.waitKey(1)==ord('s')):
                cv2.imwrite("./Saved-Filter/"+str(time.time())+".jpg",frame)
            if(cv2.waitKey(1)==27):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)