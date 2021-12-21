

# Hog Method

[Histogram of Oriented Gradients method](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) invented in 2005.

To find faces in an image, we will look at every single pixel in an image one at a time.For every single pixel, we want to look at the pixels that directly surrounding it.

The main goal is to find out which direction the image is getting darker, and then draw an arrow to that pixel.

When you repeat that action for every single pixel in the image, you will see that every pixel is being replaced by an arrow.
These arrows are gradients.

They show the flow from light to dark [in an image.](https://iq.opengenus.org/object-detection-with-histogram-of-oriented-gradients-hog/)





<img width="726" alt="Screen Shot 2021-12-20 at 15 39 11" src="https://user-images.githubusercontent.com/29928837/146768486-b2474009-5a18-4c6b-a569-e4996306f37a.png">


![hog8](https://user-images.githubusercontent.com/29928837/146944652-c65b4bc0-7107-4135-ae0c-a6bc9b40d70e.png)


```python

from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2



image = cv2.imread('/Users/randyasfandy/Desktop/face detection/nicole.jpeg')
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(1000, 500),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original image')

```

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
plt.savefig('hog10.png')


## Step 1 Finding The Faces

Locating the face

<img width="445" alt="Screen Shot 2021-12-21 at 16 41 47" src="https://user-images.githubusercontent.com/29928837/146939774-6d16e923-332e-494b-87ce-fd1baee49cf1.png">

```python

# Finding face in defined path

import cv2
import dlib

path = '/Users/randyasfandy/Desktop/face detection/nicole.jpeg'
img = cv2.imread(path)

detector = dlib.get_frontal_face_detector()
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imggray)

for face in faces :
    x1,y1 = face.left(), face.top()
    x2,y2 = face.right(), face.bottom()
    img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

cv2.imshow("Original", img)
cv2.waitKey(0)

```
## Step 2 Find the Facial Landmark

Draw a line between eyes to lips and the edge of human faces.

Sample of finding facial landmarks

<img width="439" alt="Screen Shot 2021-12-21 at 16 52 01" src="https://user-images.githubusercontent.com/29928837/146941123-42890b32-9dd2-4127-9a12-cd111d22b896.png">


```python

#Finding facial landmarks in dlib

import cv2

import numpy as np

import dlib

img = cv2.imread('/Users/randyasfandy/Desktop/nicole.jpeg')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/randyasfandy/Downloads/shape_predictor_68_face_landmarks.dat")

imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imggray)

for face in faces :
    x1,y1 = face.left(), face.top()
    x2,y2 = face.right(), face.bottom()
    landmarks = predictor(imggray, face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img,(x,y), 5, (50,50,255), cv2.FILLED)


cv2.imshow("Original", img)
cv2.waitKey(0)


```


## Step 3 Encoding Faces

![IMG_8945](https://user-images.githubusercontent.com/29928837/146953485-753bcad4-c729-4949-be9c-798e49ba0bbe.JPG)

[Photos Credit]https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78


128 Measurements Generated from an Image.



## Step 4 Find Person Name From The Encoding

![IMG_8946](https://user-images.githubusercontent.com/29928837/146953793-5d381b5d-4614-4417-921d-5409a8b2b07e.JPG)


[Photos Credit](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)

Applying machine learning classification algorithm between those measurements.

I use SVM.(Support Vector Machine)

By this it can be differentiate between measurements and it can be tell that the measurements are of the given person or not.

## Examples

I open Web Camera and then show photos from my phone to camera.

### Bill Gates

<img width="520" alt="Screen Shot 2021-12-20 at 16 21 39" src="https://user-images.githubusercontent.com/29928837/146773527-b14102ba-0f8f-440a-beb0-557eba5edfa2.png">

### Jack Ma

<img width="450" alt="Screen Shot 2021-12-20 at 16 12 08" src="https://user-images.githubusercontent.com/29928837/146773055-720c0416-2de3-47b8-94aa-48ebd87d1184.png">



## Face Comparisons

### Elon Musk & Bill Gates

<img width="719" alt="Screen Shot 2021-12-20 at 16 11 07" src="https://user-images.githubusercontent.com/29928837/146772407-08a906a1-8159-4e32-a12e-70356eea99ab.png">

### Elon Musk & Elon Musk


<img width="976" alt="Screen Shot 2021-12-20 at 16 11 21" src="https://user-images.githubusercontent.com/29928837/146772498-a6cbd4e1-d6d2-4ebc-8043-aa58250d5213.png">

## Little Bit Fun 

To surprise my beloved fiance ; 

Bitanem means my only one in my language :)

<img width="979" alt="Screen Shot 2021-12-20 at 16 11 50" src="https://user-images.githubusercontent.com/29928837/146772968-a09fe714-4e60-44b1-8624-9239d44fbc34.png">



To depict my newly skill to my father :) 

Babisko means daddy , who is my father.

<img width="366" alt="Screen Shot 2021-12-20 at 16 10 33" src="https://user-images.githubusercontent.com/29928837/146772916-fa3df573-3b85-488d-b99a-9475f44dcc00.png">






