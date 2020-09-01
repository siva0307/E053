#!/usr/bin/env python
# coding: utf-8

# In[35]:


import tensorflow as tf


# In[36]:


mnist = tf.keras.datasets.mnist # most basic dataset for deeplearning ( dataset of handwritten numbers 0-9) 28x28 image

(x_train, y_train),(x_test, y_test) = mnist.load_data() # unpacking dataset to training and testing models

x_train = tf.keras.utils.normalize(x_train, axis=1) # scaling the values from 250+ down to values between 0 and 1 to make it easier for a neural network to learn

x_test = tf.keras.utils.normalize(x_train, axis=1)
 


# In[37]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0] , cmap =plt.cm.binary) # black n white
plt.show()

print(x_train[0]) #shows its an array knowing the values of each cell





# In[38]:


#building the model
    
model = tf.keras.models.Sequential() # using the sequential model
model.add(tf.keras.layers.Flatten()) # first layer (input) image now is multidimensional, changing it to be flat
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))# hidden layers, 128 neurons, activation stage, relu: rectified linear
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))# output layer, neurons will be the number of classifications. 10 (numbers 0-9) softmax used as it is more of a probability distribution

# done defining the architecture of the model

# defining the parameters for the training of the model

model.compile(optimizer= 'adam',  #loss=degree of error  
             loss= 'sparse_categorical_crossentropy',
             metrics=['accuracy'])

#start deeplearning
model.fit(x_train, y_train,epochs=3)


# In[39]:


model.save('num_reader_model')


# In[40]:


new_model = tf.keras.models.load_model('num_reader_model')


# In[41]:


predictions = new_model.predict(x_test)
print(predictions)


# In[42]:


import numpy as np

print(np.argmax(predictions[0]))


# In[43]:


plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

