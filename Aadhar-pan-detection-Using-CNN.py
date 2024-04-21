#!/usr/bin/env python
# coding: utf-8

# Objective: Developed a CNN-based deep learning model for detecting Aadhar and PAN cards in images, employing CNN technique for accurate identification and recognition.

# # Import Libraries

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Convolution2D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score,classification_report,multilabel_confusion_matrix

import matplotlib.pyplot as plt


# # Data Collection

# In[2]:


image_size =(64, 64)

def load_images_from_folder(folder):
    images=[]
    labels=[]
    label=os.path.basename(os.path.normpath(folder))
    for filename in os.listdir(folder):
        img=cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

def data_generator(images, labels, batch_size):
    while True:
          for i in range(0, len(images), batch_size):
            batch_images=images[i:i+batch_size]
            batch_labels=labels[i:i+batch_size]
            batch_images=np.array(batch_images, dtype=np.float32) / 255.0
            batch_labels=np.array(batch_labels)

            yield batch_images, batch_labels


# # Import Data from Local File 

# In[4]:


pan_images, pan_labels =load_images_from_folder(r"E:\ds\Class_material\DL\CNN_Project_DS\Pan")
aadhar_images, aadhar_labels= load_images_from_folder(r"E:\ds\Class_material\DL\CNN_Project_DS\Aadhar")


# In[5]:


pan_images


# In[6]:


pan_labels


# # Collect all images and Labels in single folder

# In[7]:


images=pan_images + aadhar_images
labels=pan_labels + aadhar_labels


# In[8]:


images


# In[9]:


labels


# # Convert label variable in Int format

# In[10]:


label_to_int={"Pan":0,"Aadhar":1}


# In[11]:


int_labels = np.array([label_to_int[label] for label in labels])


# In[12]:


int_labels


# In[13]:


images1=np.array(images)


# # Spliting

# In[14]:


x_train, x_test, y_train1, y_test=train_test_split(images1, int_labels, test_size=0.2, random_state=42)


# In[15]:


x_train


# # Display Image

# In[16]:


A=plt.imshow(x_train[18])


# # EDA

# In[17]:


x_train.shape


# In[18]:


y_test.shape


# # Preprocessing

# In[19]:


x_train=x_train/255
x_test=x_test/255


# In[ ]:





# In[20]:


y_train=to_categorical(y_train1, num_classes=2)
y_train


# # Model

# In[21]:


nn = Sequential()


# In[22]:


nn.add(Convolution2D(filters=16,kernel_size=(3,3),input_shape=(64, 64, 3)))
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(BatchNormalization())
nn.add(Dropout(0.2))

nn.add(Convolution2D(filters=16,kernel_size=(3,3)))
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(BatchNormalization())
nn.add(Dropout(0.2))

nn.add(Flatten())

nn.add(Dense(64,activation='relu'))
nn.add(Dense(128,activation='relu'))
nn.add(Dense(64,activation='relu'))

nn.add(Dense(2,activation='softmax'))


# # Callback

# In[24]:


Early=EarlyStopping(monitor='val_loss',patience=7)


# # Compile

# In[25]:


nn.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])


# # Model Fitting

# In[26]:


hist=nn.fit(x_train,y_train,validation_split=0.2,epochs=30)


# # Evaluation

# In[27]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])


# # Prediction of Testing Set

# In[28]:


y_pred=nn.predict(x_test)


# In[29]:


y_pred


# In[30]:


y_pred1=np.argmax(y_pred,axis=-1)


# In[31]:


y_pred1


# In[32]:


y_test


# # Evaluation Of Testing

# In[33]:


acc=accuracy_score(y_pred1,y_test)
clf=classification_report(y_pred1,y_test)
cnf=multilabel_confusion_matrix(y_pred1,y_test)

print('Accuracy:',acc)
print('classification_report:\n',clf)
print('Confusion_matrix:\n',cnf)


# # Create Function to Predict Unseen Image

# In[40]:



def predict_card_type(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    
    # Reshape the image to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)
    
    # Use the model to predict labels for the image
    predictions = model.predict(image)
    
    # Convert predicted probabilities into class labels
    predicted_label = np.argmax(predictions, axis=-1)
    
    # Map integer label to class name
    class_name = "Pan" if predicted_label == 0 else "Aadhar"
    
    return class_name


# # Unseen Image Prediction Using Function

# In[41]:


image_path = "E:\ds\Class_material\DL\CNN_Project_DS\demo\photo_2024_2.jpg"  # Provide the path to your image here

predicted_class = predict_card_type(image_path, nn)

print("Predicted Card Type:", predicted_class)


# In[ ]:





# In[42]:


image_path = "E:\ds\Class_material\DL\CNN_Project_DS\demo\photo_2024.jpg"

predicted_class = predict_card_type(image_path, nn)

print("Predicted Card Type:", predicted_class)


# In[ ]:




