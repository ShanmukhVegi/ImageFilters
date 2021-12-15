import dlib
import cv2

class BW_Filter:
    def __init__(self):
        self.landmark_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/Users/shanmukhvegi/Desktop/Mini-Project/shape_predictor_68_face_landmarks.dat")
    def start_process(self):
        vid_capture=cv2.VideoCapture(0)
        vid, frame = vid_capture.read()
        rows, cols, _vid= frame.shape
        while True:
            vid, frame = vid_capture.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray Image",gray_frame)
            if(cv2.waitKey(1)==27):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)