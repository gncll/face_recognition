#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:40:43 2021

@author: randyasfandy
"""


import os

import cv2 as cv


import numpy as np

people = ['Osman Pamukoğlu', 'Benedict Cumberbatch', 'Nicola Tesla', 'Mustafa Kemal Atatürk']

DIR = r'/Users/randyasfandy/PycharmProjects/FaceRecognition/PhotosTrain'

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()


print(f'Length of the features = {len(features)}')
print(f'Length of the labels =  {len(labels)}')


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the Recognizer on the features list and the labels list

face_recognizer.save('face_trained.yml')

face_recognizer.train(features,labels)

np.save('features.npy', features)

np.save('labels.npy', labels)
