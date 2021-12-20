

# Face Recognition

Developing a face recognition algorithm with using object detection Algorithm - [haarcascade](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## What is Haarcascade?

Haarcascade is an Object Detection Algorithm used to identify faces in an image or video. 

The algorithm uses edge or line detection features which created by Paul Viola & Michael Jones in the research paper [Rapid Object Detection using a Boosted Cascade of Simple Features](https://ieeexplore.ieee.org/document/990517)


## Version Check

When using haarcascade with opencv to face, it is vital to load [opencv_contrib_python](https://pypi.org/project/opencv-contrib-python/) library to be able to use [face.LBPHFaceRecognizer_create](https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html).

Because [opencv_python](https://pypi.org/project/opencv-python/) library does not include [face.LBPHFaceRecognizer_create](https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html).

## Train-Test

Please create two folder with pictures in your working directory, first algorithm will train and then test it according to the result of train.

## Loading haarcascade

Download haarcascade in xml format in your working directory if it is possible, to download; 

1. Open it in [raw format](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml) 
2. Press Ctrl+A and then Ctrl+V or for mac Command+A and Command+V
3. Create a new file in your directory via your python interpreter.I recommend Anaconda- Spyder, because I had an issue while loading face.LBPHFaceRecongizer.create with PyCharm CE.
4. Paste into file and save it.
5. You will use it in your code.
6. In newer versions of opencv haarcascades you can install it with 
```ruby
cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

```


## Examples

### Mustafa Kemal Ataturk

<img width="620" alt="Screen Shot 2021-12-20 at 20 13 23" src="https://user-images.githubusercontent.com/29928837/146806034-84c0c3d7-490a-4159-a831-3f5fb405342d.png">


### Osman Pamukoglu

<img width="301" alt="Screen Shot 2021-12-20 at 20 13 12" src="https://user-images.githubusercontent.com/29928837/146806088-17353b0f-4658-4a6d-a645-eaffc11be860.png">


### Nicola Tesla

<img width="193" alt="Screen Shot 2021-12-20 at 20 11 38" src="https://user-images.githubusercontent.com/29928837/146806115-1564123d-bec1-4776-84ae-20d8459e88d7.png">


## Final Note

It should be important to select high definition photos to avoid misrecognition in Object Detection Algorithm method.


