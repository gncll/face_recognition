# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

import cv2 as cv


import numpy as np

people = ['Osman Pamukoglu', 'Benedict Cumberbatch', 'Nicola Tesla', 'Mustafa Kemal Ataturk']

#List of all the people in the imgae.

DIR = r'/Users/randyasfandy/PycharmProjects/FaceRecognition/PhotosTrain'

#Direction of photos

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Haar cascade

features = []

# The images array of faces.

labels = []

# Every faces in this features list, what is corresponding label, whose face does it belong to.


def create_train():
    
#That will loop over every image, and grab face in these images, add that to our training set.

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        
        #Loop over every person in the people list, grab the path for this person, 

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            
        # we are inside of each folder, we will loop over every image in that folder.

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            
        # Now we have path of image, we will read these image, and turn color from blue-green-red to gray.

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        # Detect the faces with the help of haar cascade.

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                
                features.append(faces_roi)
                labels.append(label)
                
        # Loop over every face in faces_rect and append features and labels.


create_train()


print(f'Length of the features = {len(features)}')
print(f'Length of the labels =  {len(labels)}')


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the Recognizer on the features list and the labels list.

#Attention, please load opencv_contrib_python to find face function in opencv library. 

face_recognizer.train(features,labels)



np.save('features.npy', features)

np.save('labels.npy', labels)

#Convert features and labels to numpy array

face_recognizer.save('face_trained.yml')

#Save this trained model to use it in another file.


