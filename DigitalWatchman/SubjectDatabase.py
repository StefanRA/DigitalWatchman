import os
import cv2
import numpy as np
from FaceDetector import FaceDetector

class SubjectDatabase(object):
    def __init__(self, subjectsFileName):
        self.loadSubjectNames(subjectsFileName)

    def loadSubjectNames(self, subjectsFileName):
        file = open(subjectsFileName)
        self.subjects = eval(file.read())

    def prepareTrainingData(self, trainingDataFolderPath):
        faceDetector = FaceDetector("Haar")
        faces = []
        labels = []
        directories = os.listdir(trainingDataFolderPath)
        for directory in directories:
            print(directory)
            subjectDirectoryPath = trainingDataFolderPath + "/" + directory
            subjectImageNames = os.listdir(subjectDirectoryPath)
            for subjectImageName in subjectImageNames:
                subjectImagePath = subjectDirectoryPath + "/" + subjectImageName
                subjectImage = cv2.imread(subjectImagePath)
                detectedFaces = faceDetector.detectFaces(subjectImage)
                for (x, y, w, h) in detectedFaces:
                    grayImage = cv2.cvtColor(subjectImage, cv2.COLOR_BGR2GRAY)
                    face = grayImage[y:y+w, x:x+h]
                    face = cv2.resize(face, (200, 200))
                    faces.append(face)
                    labels.append(int(directory))
        return faces, labels
