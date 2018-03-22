import cv2
from FaceDetector import FaceDetector
import numpy as np
from SubjectDatabase import SubjectDatabase

def displayPredictionsOnImage(faceRecognizer):
    faces = FaceDetector.detectFaces()

def testRecognizeFaces():
    subjectDB = SubjectDatabase("Subjects.txt")
    print(subjectDB.subjects)
    faces, labels = subjectDB.prepareTrainingData("Subjects")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save("facerec.yml")

    cam = cv2.VideoCapture(0)
    faceDetector = FaceDetector("Haar")
    while(True):
        ret,img = cam.read()
        height = np.size(img,0)
        width = np.size(img,1)
        faces = faceDetector.detectFaces(img)
        for (x, y, w, h) in faces:
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = grayImage[y:y+w, x:x+h]
            face = cv2.resize(face, (200, 200))
            label, confidence = face_recognizer.predict(face)
            print("%d %d", (label, confidence))
            cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 255), 2)
            if confidence < 100:
                draw_text(img, subjectDB.subjects[label], x, y)
            else:
                draw_text(img, "Unknown", x, y)
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

def detectFacesOnWebcamStream():
    cam = cv2.VideoCapture(0)
    faceDetector = FaceDetector("Haar")
    while(True):
	    ret, frame = cam.read()
	    faces = faceDetector.detectFaces(frame)
	    for (x, y, w, h) in faces:
		    cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 255), 2)
	    cv2.imshow("Face", frame)
	    if (cv2.waitKey(1) == ord('q')):
		    break
    cam.release()
    cv2.destroyAllWindows()

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def main():
    #detectFacesOnWebcamStream()
    testRecognizeFaces()

main()