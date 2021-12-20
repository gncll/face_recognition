

# Hog Method

[Histogram of Oriented Gradients method](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) invented in 2005.

To find faces in an image, we will look at every single pixel in an image one at a time.For every single pixel, we want to look at the pixels that directly surrounding it.

Main goal is to find out which direction the image is getting darker, and then draw an arrow to that pixel.

When you repeat that action for every single pixel in the image, you will see that every pixel being replaced by an arrow. 

These arrows are gradients.

They show the flow from light to dark in an image.


<img width="726" alt="Screen Shot 2021-12-20 at 15 39 11" src="https://user-images.githubusercontent.com/29928837/146768486-b2474009-5a18-4c6b-a569-e4996306f37a.png">



# Examples

<img width="1459" alt="BillGates" src="https://user-images.githubusercontent.com/29928837/146178672-0cea1575-47bc-4f5a-8bee-40a771dda418.png">


