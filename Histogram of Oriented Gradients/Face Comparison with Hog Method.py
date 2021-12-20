import cv2

import numpy as np

import face_recognition

#Step 1- Test and Train image
#Step 2- Finding Faces in our image and finding Encodings in that.
#Step 3- Comparing the faces and finding the distances between them. 
#We are getting the encodings and then we are comparing these measurements (128.)


imgElon = face_recognition.load_image_file('ImageBasic/Elon Musk.jpeg')

imgElon =cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

#Loading an images and converting to RGB, because the images are coming in BGR, but the library understands it as RGB, so it should be converted.


imgTest = face_recognition.load_image_file('ImageBasic/billgates.jpeg')

imgTest =cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#We will train and get the encodings of our normal image and then we are going to use our test image.



faceLoc = face_recognition.face_locations(imgElon)[0]

#Because we only send single image, we will select first element.

encodeElon = face_recognition.face_encodings(imgElon)[0]

# We will encode the face that we detected.

cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)

# Draw a rectangle, color(purple) and thickness

# When we print(faceLoc) it returns us 4 different values. These values are top,right, bottom and left.


faceLocTest = face_recognition.face_locations(imgTest)[0]

encodeElonTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]),(255,0,255),2)



results = face_recognition.compare_faces([encodeElon], encodeElonTest)

#Applying linear SVM

faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)

# The lover the distance, the better the match is.

print(results, faceDis)

cv2.putText(imgTest, f'{results},{round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

# To put results and face distances to text.


cv2.imshow('Elon Musk', imgElon)

# To see image.

cv2.imshow('elon_test', imgTest)

# To see test image.

cv2.waitKey(3000)

# To see image for 3 seconds.
