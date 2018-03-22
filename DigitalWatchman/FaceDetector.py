import cv2

class FaceDetector(object):
    def __init__(self, algorithm):
        if algorithm == "Haar":
            self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detectFaces(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(grayImage,1.3,5)