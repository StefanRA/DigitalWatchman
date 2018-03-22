import cv2
from FaceDetector import FaceDetector
import numpy as np

def displayPredictionsOnImage(faceRecognizer):
    faces = FaceDetector.detectFaces()

def testRecognizeFaces():
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train([0], np.array([0]))
    cam = cv2.VideoCapture(0)
    faceDetector = FaceDetector("Haar")
    while(True):
        ret,img = cam.read()
        faces = faceDetector.detectFaces(img)
        for (x, y, w, h) in faces:
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label, confidence = face_recognizer.predict(grayImage[y:y+w, x:x+h])
            print("%d %d", (label, confidence))
            cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow("Face", img)
        if (cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    cam = cv2.VideoCapture(0)
    faceDetector = FaceDetector("Haar")
    while(True):
	    ret,img = cam.read()
	    faces = faceDetector.detectFaces(img)
	    for (x, y, w, h) in faces:
		    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 255), 2)
	    cv2.imshow("Face", img)
	    if (cv2.waitKey(1) == ord('q')):
		    break
    cam.release()
    cv2.destroyAllWindows()

def testDetectorOnImage(imageName):
    faceDetector = FaceDetector("Haar")
    while(True):
        img = cv2.imread(imageName)
        faces = faceDetector.detectFaces(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow("Face", img)
        if (cv2.waitKey(1) == ord('q')):
            break
    cv2.destroyAllWindows()

main()