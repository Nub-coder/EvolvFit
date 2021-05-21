#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# # Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import random
import imageio
import splitfolders
import seaborn as sns
from keras.models import Model
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
import cv2
import keras_vggface
import mtcnn
from skimage.color import rgb2gray


# In[ ]:


filename='/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/dataset/images/bhuvneshwar_kumar/0389b4a1bc.jpg'


# In[ ]:


plt.imshow(plt.imread(filename))


# In[ ]:


input_folder = "/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/dataset/images"
output = "/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/processed_data"
splitfolders.ratio(input_folder, output, seed=24, ratio=(0.6, 0.1, 0.3))


# In[ ]:


from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN


# # Using MTCNN in Extract Face Function

# In[ ]:


def extract_face(arr, required_size=(224, 224)):
    
    pixels = arr
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if(len(results)==0):
        return cv2.resize(arr.astype('uint8'),required_size)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    x2 = min(223,x2)
    y2=min(223,y2)
    face = pixels[y1:y2, x1:x2]
    try:
        face_array=cv2.resize(face,required_size)
    except:
        return cv2.resize(arr.astype('uint8'),required_size)
    return face_array

pixels = extract_face(cv2.resize(plt.imread(filename).astype('uint8'),(224,224)))
plt.imshow(pixels)

plt.show()


# In[ ]:


from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import decode_predictions


# In[ ]:


train_data_dir = "/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/processed_data/train"
valid_data_dir = "/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/processed_data/val"
test_data_dir = "/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/processed_data/test"


# In[ ]:


batch_size=8
img_height,img_width = (224,224)


# # Self Designed Pre-Processing Function

# In[ ]:


def sumit_function(arr):
    #print(type(arr[0,0,0]))
    arr = extract_face ( cv2.resize(arr.astype('uint8') ,(224,224)).astype('uint8') )
    arr = preprocess_input(arr.astype('float32'))
    return arr
plt.imshow(sumit_function(plt.imread(filename).astype('float64')))


# # Data Generator

# In[ ]:


train_datagen = ImageDataGenerator(preprocessing_function=sumit_function,
                                  shear_range=0.1,
                                  zoom_range=0.1,
                                  rotation_range=5)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_height,img_width),
batch_size=batch_size,
class_mode='categorical')

valid_generator = train_datagen.flow_from_directory(
valid_data_dir , target_size=(img_height,img_width),
batch_size=1,
class_mode='categorical')


test_generator = train_datagen.flow_from_directory(
test_data_dir,
target_size=(img_height,img_width),
batch_size=1,
class_mode='categorical')


# In[ ]:


from keras.optimizers import Adam


# # Loading the Model

# In[ ]:


base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')


for layer in base_model.layers:
    layer.trainable = False
#base_model.summary()
x = base_model.output
x = Dense (15, activation='softmax')(x)

model = Model( base_model.input, x)

model.compile(optimizer = "adam",loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.summary()


# In[ ]:


train_generator[0]


# # Fitting the Model

# In[ ]:


results = model.fit_generator(train_generator,validation_data=valid_generator,steps_per_epoch=train_generator.samples//train_generator.batch_size ,validation_steps=valid_generator.samples//valid_generator.batch_size,epochs=20)


# # Saving the Model

# In[ ]:


model.save('/Users/gopalkrishnasingh/Desktop/Projects/EvolvFit/report2.h5')


# In[ ]:


df = pd.DataFrame(model.history.history)
df.to_csv('Result.csv')


# # Prediction

# In[ ]:


prediction=model.predict_generator(test_generator,steps=len(test_generator.classes))


# In[ ]:


pred=prediction.argmax(axis=-1)


# In[ ]:


from sklearn.metrics import classification_report


# # Classification report

# In[ ]:


print(classification_report(test_generator.classes,pred))

