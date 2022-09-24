# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model
![neural](https://user-images.githubusercontent.com/74660507/192100918-83afb214-2f34-493b-aa27-42214fa59ff2.png)



## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict



## PROGRAM
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

np.unique(y_test)

model = keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="categorical_crossentropy", metrics='accuracy',optimizer="adam")

model.fit(X_train_scaled ,y_train_onehot, epochs=2,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

pd.DataFrame(model.history.history).plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

confusion_matrix(y_test,x_test_predictions)

print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![191758877-9c3798a4-7486-4b42-914d-25f3f5c52a5e](https://user-images.githubusercontent.com/74660507/192100830-4f278b3b-2e89-4bcb-8f5a-68219b9a5c6d.png)


### Classification Report
![191760299-ece89d06-91ca-451e-a368-9b12f65495ee](https://user-images.githubusercontent.com/74660507/192100836-ab1bfb33-e079-4d99-bc08-4a964a57251f.png)



### Confusion Matrix
![191760361-325a0809-4600-49c5-bfc1-75a8901ca91b](https://user-images.githubusercontent.com/74660507/192100843-828fb104-b072-432d-8090-1f763f71d002.png)



### New Sample Data Prediction
![191760417-3156388b-d5f7-4be5-a537-42f8346f05f8](https://user-images.githubusercontent.com/74660507/192100852-31349b9e-b691-43c1-815a-e025947962d8.png)



## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
