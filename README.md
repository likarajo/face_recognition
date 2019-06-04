# Face Recognition

## Libraries used

* ***numpy***: for arrays/matrices
* ***cv2***: computer vision library

## Image Recording

* Source camera location is provided and the camera object is prepared to capture images from the source camera 
* Image frames are captured using the camera
* The image frames are converted from BGR to grayscale
* Objects of different sizes are detected from the input image and returned as a list using ***Haar-CascadeClassifier***
* For each face object we have the corner coordinates (x, y) and the width and height of the object
  * Get the component from the captured frame
  * The component is transform into a data by resizing into a data by resizing
  * Store the face data after every 10 frames, till we get 20 entries
  * Render a rectangle around the face in the frame for vizualization
  * Display the image frame in a window
  * If user presses the 'Esc' key or the number of images hits 20, then recording is stopped
* The source camera is turned off and all the image frame windows created are destroyed
* The data is saved as a numpy matrix in an encoded format

## Image Recognizing

* Source camera location is provided and the camera object is prepared to capture images from the source camera 
* Saved image data files are loaded and train data set is created
* Labels are marked accordingly and a dictionary is created to map labels to the corresonding text string
* A Function to calculate ***Euclidean distance*** is defined
* ***K Nearest Neighbors*** algorithm to detect label is defined
* Image frames are captured using the camera
* The image frames are converted from BGR to grayscale
* Objects of different sizes are detected from the input image and returned as a list using ***Haar-CascadeClassifier***
* For each face object we have the corner coordinates (x, y) and the width and height of the object
  * Get the component from the captured frame
  * The component is transform into a data by resizing into a data by resizing
  * The label of the data is predicted using KNN
  * The detected label is mapped to its corresponding text which is renedered on the frame
  * A rectangle is also rendered around the face in the frame for vizualization
  * The image frame is displayed in a window
  * If user presses the 'Esc' key, then recording is stopped
* The source camera is turned off and all the image frame windows created are destroyed

