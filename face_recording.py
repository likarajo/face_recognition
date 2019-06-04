"""@author: likarajo"""import numpy as npimport cv2# Source Camera location#source = "rtsp://admin:admin2016@192.168.1.2//Streaming/Channels/2"source = 0 # PC camera# Camera object to capture imagescamera = cv2.VideoCapture(source)# Haar Cascade Classifier object for face object detectionhaarCascadeClf = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')# Font objectfont = cv2.FONT_HERSHEY_SIMPLEXdata = [] # placeholder for storing the dataix = 0    # current frame numberprint('Recording Image...')while True:    # Capture image frame    ret, frame = camera.read()        # If camera is working fine, proceed    if ret == True:                # Convert the current frame to grayscale        grayFaceImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Detect objects of different sizes from the input image and return as a list of rectangles.        # Using Haar-CascadeClassifier -> detectMultiScale(image, scaleFactor, minNeighbors)        faceObjects = haarCascadeClf.detectMultiScale(grayFaceImg, 1.3, 5)        # For each face object we have the corner coordinates (x, y) and the width and height of the object        for (x, y, w, h) in faceObjects:            # Get the component from the captured frame            face_component = frame[y:y+h, x:x+w, :]            # Transform the component into a data by resizing            face_data = cv2.resize(face_component, (50, 50))            # Store the face data after every 10 frames, till we get 20 entries            if ix%10 == 0 and len(data) < 20:                data.append(face_data)                print('Image data saved '+ str(len(data)))                            cv2.putText(frame, 'Frame #'+str(ix), (x, y-10), font, 1 , (255, 255, 0), 2)            # Render a rectangle around the face in the frame for vizualization            # rectangle(img, vertex, opp_vertex, color(BGR), thickness)            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)                ix += 1 # increment the current frame number              # Display the image frame in a window        cv2.imshow('Image Frame', frame)        # If user presses the 'Esc' key (id: 27) or the number of images hits 20, then stop recording with a delay of 1ms        if cv2.waitKey(1) == 27 or len(data) >= 20:            print('Image recorded')            break        # If the camera is not working, print "error"    else:        print("Camera Error")# now we destroy the windows we have createdcamera.release()cv2.destroyAllWindows()# convert the data to a numpy formatdata = np.asarray(data)# save the data as a numpy matrix in an encoded formatnp.save('./data/face_01', data)print ('Image data saved', data.shape)exit()