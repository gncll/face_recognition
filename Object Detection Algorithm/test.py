#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:26:55 2021

@author: randyasfandy
"""

import numpy as np

import cv2 as cv

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


people = ['Osman Pamukoğlu', 'Benedict Cumberbatch', 'Nicola Tesla', 'Mustafa Kemal Atatürk']
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('/Users/randyasfandy/PycharmProjects/FaceRecognition/face_trained.yml')

img = cv.imread(r'/Users/randyasfandy/PycharmProjects/FaceRecognition/PhotosTest/cumberbatch.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray [y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label ={people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)





