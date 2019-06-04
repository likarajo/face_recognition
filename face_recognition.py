"""
@author: likarajo
"""

import numpy as np 
import cv2

# Source Camera location
#source = "rtsp://admin:admin2016@192.168.1.2//Streaming/Channels/2"
source = 0 # PC camera

# Load image data files
face_01 = np.load('./data/face_01.npy').reshape(20, 50*50*3)
face_02 = np.load('./data/face_02.npy').reshape(20, 50*50*3)

# Create dataset
data = np.concatenate([face_01, face_02])

# Prepare labels
labels = np.zeros((40, 1))
labels[:20, :] = 0.0     # Label 0 => Raj
labels[20:40, :] = 1.0    

# Create dictionary to map labels to text
faces = {
    0: 'Raj',
    1: 'Surajit'
}        

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(sum((x1-x2)**2))

# K Nearest Neighbors algorithm to detect label 
def knn(x, train, targets, k=5):
    n = train.shape[0]
    dist = []
    for i in range(n):
        dist.append(euclidean_distance(x, train[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

# Haar Cascade Classifier object for face object detection
haarCascadeClf = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Font object
font = cv2.FONT_HERSHEY_SIMPLEX

# Camera object to capture images
camera = cv2.VideoCapture(source)

print('Detecting Image...')

while True:
    # Capture image frame
    ret, frame = camera.read()
    
    # If camera is working fine, proceed
    if ret == True:
        
        # Convert the current frame to grayscale
        grayFaceImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects of different sizes from the input image and return as a list of rectangles.
        # Using Haar-CascadeClassifier -> detectMultiScale(image, scaleFactor, minNeighbors)
        faceObjects = haarCascadeClf.detectMultiScale(grayFaceImg, 1.3, 5)
        
        # For each face object we have the corner coordinates (x, y) and the width and height of the object
        for (x, y, w, h) in faceObjects:
            
            # Get the component from the captured frame
            face_component = frame[y:y+h, x:x+w, :]
            
            # Transform the component into a data by resizing
            face_data = cv2.resize(face_component, (50, 50))
            
            # Predict the label of the data using knn
            face_label = knn(face_data.flatten(), data, labels)
            
            # Map the label to its corresponding text
            text = faces[int(face_label)]
            
            # Render the specified text string in the image
            # putText(img, text, origin, fontFace, fontScale, color(BGR)) 
            cv2.putText(frame, text, (x, y-10), font, 1 , (255, 255, 0), 2)
            
            # Render a rectangle around the face in the frame for vizualization
            # rectangle(img, vertex, opp_vertex, color, thickness) 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Display the image frame in a window
        cv2.imshow('Image Frame', frame)
        
        # If user presses the 'Esc' key (id: 27) then stop with a delay of 1ms
        if(cv2.waitKey(1) == 27):
            print('End')
            break
    
    # If the camera is not working, print "error"
    else:
        print("Camera error")

# Destroy all of the opened HighGUI windows
camera.release()
cv2.destroyAllWindows()

exit()

