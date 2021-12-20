import cv2
import numpy as np
import face_recognition
import os

# We will get the images from our folder automatically and then It will generate encodings for it and try to find it in our webcam.

path = 'Image Attendance'

# Defining Path

images = []

# We will create a blank list that all the images we will import.

classNames = []

# Names of all these images. When we output the results we will use these names.

myList = os.listdir(path)


# We will grab the list of images in this folder.

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # To remove file type.(jpg,jpeg, png)

print(classNames)

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Convert BGR to RGB.
        encode  = face_recognition.face_encodings(img)[0]
        #Find the encoding.
        encodeList.append(encode)
        #Append the list

    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding complete')


cap = cv2.VideoCapture(0)

#Initilaze the webcam.

while True:

#To get each frame one by one.

    success, img = cap.read()
    #This will give us our image
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) 
    #Reduce the size of our image, this will help us increase the process.
    #1/4 of the size (0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    facesCurFrame = face_recognition.face_locations(imgS)
    #faces in our current frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    #encodings of current frame.

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        #One by one it will grab one face location from facesCurFrame list and then it will grab the encoding of encodeface from encodesCurFrame
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # Compare the faces between encodeListKnow and encodeFace.
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # Find the distance from webcame to defined faces.
        
        #It will return list to us. Lowest distance is our best match. 
        
        #Because we are sending in a list to our faceDis function. It will return us a list.It will give as 'n' values. N is the number the faces
        #that you predefine. The lowest number of the result is matching people.(Lowest number between two picture means they are alike.)

        print(faceDis)

        matchIndex = np.argmin(faceDis)
        #Finding lowest faceDis
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 
            # To increasing the size of rectangle between face location.
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
            # 0,255,0 is color, 2 is thickness.
            
            cv2.rectangle(img,(x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            
            #Drawing another rectangle.
            
            cv2.putText(img, name, (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
           

    cv2.imshow('Webcam', img)

    cv2.waitKey(1)


