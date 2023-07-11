#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow import keras


# In[3]:


fashion_mnist = keras.datasets.fashion_mnist


# In[4]:


(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()


# In[5]:


X_train_full.shape


# In[6]:


X_valid, X_train = X_train_full[:5000]/255.0,X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000],y_train_full[5000:]


# In[7]:


class_names = ["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# In[8]:


class_names[y_train[0]]


# In[9]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))


# In[10]:


model.summary()


# In[11]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])


# In[12]:


history = model.fit(X_train,y_train,epochs = 30,validation_data=(X_valid,y_valid))


# In[13]:


import pandas as pd


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[16]:


model.evaluate(X_test,y_test)


# In[17]:


X_new = X_test[:3]


# In[19]:


y_prob = model.predict(X_new)


# In[20]:


y_prob.round(2)


# In[21]:


y_new = y_test[:3]
y_new


# In[22]:


y_new[0]


# In[ ]:




