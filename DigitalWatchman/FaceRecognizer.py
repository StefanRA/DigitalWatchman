import cv2
from FaceDetector import FaceDetector

class FaceRecognizer(object):
    def __init__(self, algorithm):
        self.faceDetector = FaceDetector("Haar")

    def recognizeFaces(self, image):
        faces = self.faceDetector.detectFaces(image)
        for face in faces:
            label, confidence = face_recognizer.predict(face)
            for subjectName, subjectLabel in subjects.items():
                if subjectLabel == label:
                    label_text = subjectName
    
            drawRectangle(img, rect)
            drawText(img, label_text, rect[0], rect[1]-5)

    def drawRectangle(image, rect):
        (x, y, w, h) = rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    def drawText(image, text, x, y):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)