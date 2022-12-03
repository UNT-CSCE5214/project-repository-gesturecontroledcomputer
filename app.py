#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import h5py
from imutils import paths
import imutils
from PIL import Image
import keyboard


# In[2]:


import cv2
import imutils

class Movement_dectector:

    def __init__(self, accumulated_weights=0.5):
        
        # Accumulated weight factor
    
        self.accumulated_weights = accumulated_weights

        # Background model

        self.bgrd = None

    

    def detect_img(self, image, tVal=25):

        # Absloute difference between background model and image
        # passed in, then threshold the delta image
        delta = cv2.absdiff(self.bgrd.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # return none if no contours are present

        if len(cnts) == 0:
            return None

        # returns the tuple threshold image

        return (thresh, max(cnts, key=cv2.contourArea))

    def update_bgrd(self, image):
   
        if self.bgrd is None:
            self.bgrd = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, self.bgrd, self.accumulated_weights)


# In[3]:


hand_positions=['back','close','enter','maximize','minimize','next','previous','spacebar','volumedown','volumeup'] 

key_press = ['backspace','alt+f4','enter','windows+up','windows+down','right','left','space','Volume_Down','Volume_UP']

# loading the json and creating the model

json_file = open("model2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

print("Module loaded")

camera = cv2.VideoCapture(0)
ROI = "10,350,225,590"

(top, right, bot, left) = np.int32(ROI.split(","))
md = Movement_dectector()
no_of_frames = 0
consec = 0
previous = "none"
pred = "nil"
max_indexes = []
max_indexes.append("0")

# Loop until the "q" is pressed

while True:
    # Current frame
    (grabbed_img, fetched_frame) = camera.read()

    fetched_frame = cv2.flip(fetched_frame, 1)
    
    #Resizing the fetched_frame
    
    fetched_frame = imutils.resize(fetched_frame, width=600)
    
    clone = fetched_frame.copy()
    (frameH, frameW) = fetched_frame.shape[:2]

    
    roi = fetched_frame[top:bot, right:left]
    hand = roi.copy()
    
    # Converting to the Gray
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if no_of_frames < 200:
        md.update_bgrd(gray)

    
    else:
        # Detecting the skin image
        
        skin = md.detect_img(gray)


        # IF skin is detected
        
        if skin is not None:
            
            (thresh, c) = skin
            
            (_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            c = max(cnts, key=cv2.contourArea)
            
            (x, y, w, h) = cv2.boundingRect(c)
            
            logo = thresh[y:y + h, x:x + w]
            
            logo = cv2.resize(logo, (100, 100))
            logo =  cv2.cvtColor(logo, cv2.COLOR_GRAY2RGB)
            im_pil = Image.fromarray(logo)
            img_pred = image.img_to_array(im_pil)
            img_pred = np.expand_dims(img_pred, axis = 0)
            rslt = loaded_model.predict(img_pred)
            A = np.array(rslt[0])
            max_indexes = np.where(A==max(rslt[0]))
            pred = str(hand_positions[int(max_indexes[0][0])])
            cv2.putText(clone, pred, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 2)
            cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
            cv2.imshow("Threshed Images", thresh)


    # draw the hand ROI and increment the number of processed frames
    
    cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
    if (previous==pred):
        consec+=1

    previous=pred

    if (consec>50):
        print (consec)
        keyboard.press_and_release(str(key_press[int(max_indexes[0][0])]))
        print(str(key_press[int(max_indexes[0][0])]))
        consec=0
    no_of_frames += 1
    if no_of_frames >= 30:
        if fl ==1:
            print ("Calibration Completed")
            fl=0
    else :
        print (no_of_frames)
        fl = 1

    
    cv2.imshow("fetched_frame", clone)
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, loop will be breaked
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


# In[ ]:




