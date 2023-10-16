import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()


# In[3]:


X_train_full, X_test = X_train_full / 255.0, X_test / 255.0

# Split the dataset into training, validation, and test sets
X_train, X_valid = X_train_full[:45000], X_train_full[45000:]
y_train, y_valid = y_train_full[:45000], y_train_full[45000:]


# In[4]:


class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[5]:


num_models = 5
models = []


# In[6]:


data_augmentation = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0,
    shear_range=0,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[7]:


model_histories = []


# In[ ]:


for i in range(num_models):
    model = keras.models.Sequential()

    if i == 0:
                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
                model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(512, activation='relu'))
                model.add(keras.layers.Dense(10, activation='softmax'))
                  
    elif i == 1:
                model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(256, activation='relu'))
                model.add(keras.layers.Dense(10, activation='softmax'))
    elif i == 2:

                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(128, activation='relu'))
                model.add(keras.layers.Dense(10, activation='softmax'))
                          
    elif i == 3:

                model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(256, activation='relu'))
                model.add(keras.layers.Dense(10, activation='softmax'))
                          
    elif i == 4:
    
                model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(512, activation='relu'))
                model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))
    model_histories.append(history)
    models.append(model)
    


# In[ ]:


for i in range(num_models):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model_histories[i].history['loss'], label='Train')
    plt.plot(model_histories[i].history['val_loss'], label='Validation')
    plt.title('Model {} Loss'.format(i+1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model_histories[i].history['accuracy'], label='Train')
    plt.plot(model_histories[i].history['val_accuracy'], label='Validation')
    plt.title('Model {} Accuracy'.format(i+1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


# In[ ]:


ensemble_predictions = []
for model in models:
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=-1)
    ensemble_predictions.append(y_pred)


# In[ ]:


import numpy as np


# In[ ]:


ensemble_predictions = np.array(ensemble_predictions)
final_predictions = np.median(ensemble_predictions, axis=0).astype(int)


# In[ ]:


ensemble_accuracy = np.mean(final_predictions == y_test.flatten())
print("Ensemble Accuracy:", ensemble_accuracy)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


ensemble_probabilities = [model.predict(X_test) for model in models]
ensemble_probabilities = np.mean(ensemble_probabilities, axis=0)
roc_auc = roc_auc_score(y_test, ensemble_probabilities, multi_class='ovr')

print("Ensemble ROC-AUC Score:", roc_auc)


# In[ ]:


from sklearn.metrics import recall_score, f1_score


# In[ ]:


ensemble_predictions = np.median(ensemble_predictions, axis=0).astype(int)  # Use the ensemble predictions
recall = recall_score(y_test, ensemble_predictions, average='weighted')
f1 = f1_score(y_test, ensemble_predictions, average='weighted')

print("Ensemble Recall:", recall)
print("Ensemble F1-Score:", f1)


import pickle

model_pkl_file = "cifa.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)





