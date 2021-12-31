![yakıt tüketimi](https://user-images.githubusercontent.com/29928837/146723276-b0d8325c-b43c-492d-9fa0-56aaef890826.png)


# Face Recognition
Developing a face recognition algorithm in two famous methods.

## 1. HOG(Histogram of Gradient) 

Navneet Dalal and Bill Triggs study the question sets for robust visual object recognition, adopting linear SVM based human detection as a test case.The show experimentally that grids of Histograms of Oriented Gradient (HOG) descriptors significantly outperform existing feature sets for human detection.
[HOG](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

![hog7](https://user-images.githubusercontent.com/29928837/146932969-d9f36742-7e5d-4344-abd9-059a8b0659dc.png)

To turn your photo to Hog image;

```python
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2


image = cv2.imread('/Users/randyasfandy/Downloads/Adsız tasarım.png')
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(100, 100),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
plt.savefig('hog7.png')

```



## 2. Object Detection Method

The algorithm uses edge or line detection features which created by Paul Viola & Michael Jones [in the research paper](https://ieeexplore.ieee.org/document/990517)

<img width="222" alt="Screen Shot 2021-12-20 at 19 52 35" src="https://user-images.githubusercontent.com/29928837/146803343-eaa34da4-9a50-4f34-b9e4-0bf7eb8d716f.png">

[Picture from](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)



