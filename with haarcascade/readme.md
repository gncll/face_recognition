
# Face Recognition

Developing a face recognition algorithm with using [haarcascade](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## Haarcascade

Haarcascade is an Object Detection Algorithm used to identify faces in an image or video. 

The algorithm uses edge or line detection features which created by Paul Viola & Michael Jones in the research paper [Rapid Object Detection using a Boosted Cascade of Simple Features](https://ieeexplore.ieee.org/document/990517)



### Note

When using haarcascade with opencv to face, it is vital to load [opencv_contrib_python](https://pypi.org/project/opencv-contrib-python/) library to be able to use [face.LBPHFaceRecognizer_create](https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html).

Because [opencv_python](https://pypi.org/project/opencv-python/) library does not include [face.LBPHFaceRecognizer_create](https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html).

### Note

Please create two folder in your working directory, first algorithm will train and then test it according to the result of train.

### Note

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

<img width="1332" alt="Screen Shot 2021-12-20 at 11 41 35" src="https://user-images.githubusercontent.com/29928837/146737945-8b1772f6-0f77-43a1-a39c-91e8d69bbc50.png">


### Osman Pamukoglu

<img width="603" alt="Screen Shot 2021-12-20 at 11 32 24" src="https://user-images.githubusercontent.com/29928837/146736975-6fbfb813-6937-4ab3-9924-bb75aa1baa9a.png">

### Nicola Tesla

<img width="391" alt="Screen Shot 2021-12-20 at 11 37 50" src="https://user-images.githubusercontent.com/29928837/146737510-2eba6cd7-726e-4ce9-9445-87191fdbcfde.png">



