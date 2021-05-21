#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


# In[ ]:


from numpy import asarray
from mtcnn.mtcnn import MTCNN


# In[ ]:


model = load_model('/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/report2.h5')


# In[ ]:


lst=['bhuvneshwar_kumar', 'dinesh_karthik', 'hardik_pandya', 'jasprit_bumrah', 'k._l._rahul', 'kedar_jadhav', 'kuldeep_yadav', 'mohammed_shami', 'ms_dhoni', 'ravindra_jadeja', 'rohit_sharma', 'shikhar_dhawan', 'vijay_shankar', 'virat_kohli', 'yuzvendra_chahal']


# In[ ]:


def extract_face(arr, required_size=(224, 224)):
    pixels = arr
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if(len(results)==0):
        return (-1,-1,-1,-1)
    
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    return (x1,y1,width,height)


# In[ ]:


from PIL import Image


# # Code To Open the Camera

# In[ ]:


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    labels=[]
    faces=[extract_face(frame)]

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if(x==-1):
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            x2 = x+w
            y2 = y+h
            face = frame[y:y2, x:x2]
            image = Image.fromarray(face)
            image = image.resize((224,224))
            face_array = asarray(image)
            preds = model.predict(face_array.reshape(1,224,224,3)).argmax()
            label = lst[preds]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Player_Matching',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

