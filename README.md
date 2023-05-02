# Mango Image Classification using VGG16

This repository contains a Python code for image classification of mangoes using the VGG16 model. The aim of the project is to classify the type of mango and its quality based on the image of the mango.

## Dataset
The dataset used for this project contains images of different types of mangoes, including Chasuna, Sindhri, and Anwar Ratool. The dataset also contains images of mangoes of different qualities, such as good quality as GRADE A, average quality as GRADE B, and poor quality as GRADE C.

## VGG16 Model
The VGG16 model is a convolutional neural network that was trained on a large image dataset called ImageNet. The model has 16 layers, including 13 convolutional layers and 3 fully connected layers. It is a widely used model for image classification tasks due to its high accuracy.

## Requirements

* Python 3.x
* Keras
* Tensorflow
* OpenCV
* NumPy
## Usage
Clone this repository: git clone [https://github.com/rafaymahmood1/vgg16-type-grade-classification.git](https://github.com/RafayMahmood1/VGG16-Type-Grade-Classification.git)
Run the Notebook-VGG16.ipynb file to train the model
## Results
After training the model on the dataset, the accuracy achieved on the test set was around 90%. The model was able to correctly classify the type of mango and its quality based on the image of the mango.

```python
import numpy as np
import pandas as pd
from pathlib import Path
import os.path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import time
```


```python

```


```python
image_dir = Path(r'/content/drive/MyDrive/HUMZA_FYP/Classification_dataset')

```


```python
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))


filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

category_samples = []
for category in images['Label'].unique():
    category_slice = images.query("Label == @category")
    category_samples.append(category_slice.sample(100, random_state=1))
image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

batch_size = 16
img_height = 228
img_width = 228
channels = 3
img_shape = (img_height, img_width, channels)

# pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')
pre_trained = VGG16(input_shape = img_shape,include_top = False, weights = 'imagenet')


for layer in pre_trained.layers:
    layer.trainable = False
    
num_c = 8


# x = pre_trained.output
# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
# x = Dropout(0.2)(x)
# x = Dense(1024, activation='LeakyReLU')(x)
# x = Dropout(0.2)(x)
# predictions = Dense(num_c, activation='softmax')(x)

# model1 = Model(inputs = pre_trained.input, outputs = predictions)
# model1.compile(optimizer = Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


x = layers.Flatten()(pre_trained.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(8, activation='softmax')(x)
model1 = tf.keras.models.Model(pre_trained.input, x)

model1.compile(optimizer = Adam(lr=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])


STEP_SIZE_TRAIN = train_images.n // train_images.batch_size
STEP_SIZE_VALID = val_images.n // val_images.batch_size

history = model1.fit_generator(train_images,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = val_images,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = 15,
                    verbose = 1)

results = model1.evaluate(test_images, verbose=0)
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
```

    Found 448 validated image filenames belonging to 8 classes.
    Found 112 validated image filenames belonging to 8 classes.
    Found 240 validated image filenames belonging to 8 classes.
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58889256/58889256 [==============================] - 0s 0us/step


    WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.
    <ipython-input-9-e3d2a5041c1f>:88: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      history = model1.fit_generator(train_images,


    Epoch 1/15
    14/14 [==============================] - 161s 11s/step - loss: 6.9891 - acc: 0.2054 - val_loss: 1.5297 - val_acc: 0.5104
    Epoch 2/15
    14/14 [==============================] - 6s 440ms/step - loss: 1.7352 - acc: 0.5112 - val_loss: 0.8965 - val_acc: 0.7500
    Epoch 3/15
    14/14 [==============================] - 6s 439ms/step - loss: 0.7812 - acc: 0.7701 - val_loss: 0.5456 - val_acc: 0.8750
    Epoch 4/15
    14/14 [==============================] - 6s 411ms/step - loss: 0.5781 - acc: 0.8192 - val_loss: 0.4792 - val_acc: 0.8646
    Epoch 5/15
    14/14 [==============================] - 7s 518ms/step - loss: 0.4772 - acc: 0.8504 - val_loss: 0.3495 - val_acc: 0.9271
    Epoch 6/15
    14/14 [==============================] - 6s 403ms/step - loss: 0.3446 - acc: 0.8951 - val_loss: 0.3113 - val_acc: 0.9271
    Epoch 7/15
    14/14 [==============================] - 7s 477ms/step - loss: 0.2326 - acc: 0.9464 - val_loss: 0.2491 - val_acc: 0.9479
    Epoch 8/15
    14/14 [==============================] - 6s 405ms/step - loss: 0.1905 - acc: 0.9598 - val_loss: 0.1795 - val_acc: 0.9583
    Epoch 9/15
    14/14 [==============================] - 6s 412ms/step - loss: 0.2060 - acc: 0.9487 - val_loss: 0.2905 - val_acc: 0.9271
    Epoch 10/15
    14/14 [==============================] - 7s 472ms/step - loss: 0.1532 - acc: 0.9732 - val_loss: 0.2149 - val_acc: 0.9479
    Epoch 11/15
    14/14 [==============================] - 6s 412ms/step - loss: 0.1336 - acc: 0.9754 - val_loss: 0.1971 - val_acc: 0.9375
    Epoch 12/15
    14/14 [==============================] - 7s 495ms/step - loss: 0.1169 - acc: 0.9799 - val_loss: 0.2557 - val_acc: 0.9375
    Epoch 13/15
    14/14 [==============================] - 6s 415ms/step - loss: 0.1260 - acc: 0.9710 - val_loss: 0.2633 - val_acc: 0.9271
    Epoch 14/15
    14/14 [==============================] - 10s 735ms/step - loss: 0.1294 - acc: 0.9732 - val_loss: 0.2530 - val_acc: 0.9271
    Epoch 15/15
    14/14 [==============================] - 9s 653ms/step - loss: 0.0931 - acc: 0.9866 - val_loss: 0.2440 - val_acc: 0.9271
    Test Accuracy: 96.25%



```python
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label='test set')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fbcbfb712a0>




    
![png](output_6_1.png)
    



```python
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['acc'], label='training set')
plt.plot(history.history['val_acc'], label='test set')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fbcb045fee0>




    
![png](output_7_1.png)
    



```python
predictions = np.argmax(model1.predict(test_images), axis=1)

cm = confusion_matrix(test_images.labels, predictions)
clr = classification_report(test_images.labels, predictions, 
                            target_names=test_images.class_indices, 
                            zero_division=0)
```

    8/8 [==============================] - 3s 300ms/step



```python
plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(num_c) + 0.5, labels=test_images.class_indices, rotation=90)
plt.yticks(ticks=np.arange(num_c) + 0.5, labels=test_images.class_indices, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```


    
![png](output_9_0.png)
    



```python
# from keras.models import load_model

# model1.save('model_1_v2.h5')
```


```python
image_dir = Path(r'/content/drive/MyDrive/HUMZA_FYP/Grading Dataset')

filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

category_samples = []
for category in images['Label'].unique():
    category_slice = images.query("Label == @category")
    category_samples.append(category_slice.sample(200, random_state=1))
image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)
```


```python
strr = str(train_df['Filepath'].iloc[2])
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(strr)
imgplot = plt.imshow(img)
plt.show()
```


    
![png](output_13_0.png)
    



```python


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)


train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

batch_size = 16
img_height = 228
img_width = 228
channels = 3
img_shape = (img_height, img_width, channels)


# pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')
pre_trained = VGG16(input_shape = img_shape,include_top = False, weights = 'imagenet')


for layer in pre_trained.layers:
    layer.trainable = False
    
num_c = 3

# x = pre_trained.output
# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
# x = Dropout(0.2)(x)
# x = Dense(1024, activation='LeakyReLU')(x)
# x = Dropout(0.2)(x)
# predictions = Dense(num_c, activation='softmax')(x)

# model2 = Model(inputs = pre_trained.input, outputs = predictions)
# model2.compile(optimizer = Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

x = layers.Flatten()(pre_trained.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(3, activation='softmax')(x)
model2 = tf.keras.models.Model(pre_trained.input, x)

model2.compile(optimizer = Adam(lr=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])


STEP_SIZE_TRAIN = train_images.n // train_images.batch_size

STEP_SIZE_VALID = val_images.n // val_images.batch_size

history = model2.fit_generator(train_images,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = val_images,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = 10,
                    verbose = 1)

results = model2.evaluate(test_images, verbose=0)
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
```

    Found 336 validated image filenames belonging to 3 classes.
    Found 84 validated image filenames belonging to 3 classes.
    Found 180 validated image filenames belonging to 3 classes.


    WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.
    <ipython-input-21-31e86d792e6e>:87: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      history = model2.fit_generator(train_images,


    Epoch 1/10
    10/10 [==============================] - 81s 8s/step - loss: 6.6758 - acc: 0.3520 - val_loss: 1.4054 - val_acc: 0.6250
    Epoch 2/10
    10/10 [==============================] - 3s 343ms/step - loss: 2.2928 - acc: 0.5000 - val_loss: 0.4527 - val_acc: 0.7812
    Epoch 3/10
    10/10 [==============================] - 4s 438ms/step - loss: 0.8853 - acc: 0.7072 - val_loss: 0.3713 - val_acc: 0.8750
    Epoch 4/10
    10/10 [==============================] - 4s 419ms/step - loss: 0.5078 - acc: 0.8125 - val_loss: 0.4090 - val_acc: 0.7969
    Epoch 5/10
    10/10 [==============================] - 3s 343ms/step - loss: 0.3204 - acc: 0.8750 - val_loss: 0.3608 - val_acc: 0.8281
    Epoch 6/10
    10/10 [==============================] - 6s 624ms/step - loss: 0.2917 - acc: 0.8947 - val_loss: 0.3140 - val_acc: 0.8438
    Epoch 7/10
    10/10 [==============================] - 3s 345ms/step - loss: 0.2282 - acc: 0.9145 - val_loss: 0.3211 - val_acc: 0.9062
    Epoch 8/10
    10/10 [==============================] - 3s 347ms/step - loss: 0.2404 - acc: 0.9276 - val_loss: 0.2924 - val_acc: 0.8906
    Epoch 9/10
    10/10 [==============================] - 5s 464ms/step - loss: 0.2130 - acc: 0.9375 - val_loss: 0.3101 - val_acc: 0.8594
    Epoch 10/10
    10/10 [==============================] - 4s 403ms/step - loss: 0.1773 - acc: 0.9605 - val_loss: 0.2754 - val_acc: 0.9062
    Test Accuracy: 90.56%



```python
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label='test set')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fbc9c2ade40>




    
![png](output_15_1.png)
    



```python
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['acc'], label='training set')
plt.plot(history.history['val_acc'], label='test set')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fbc9c33ffa0>




    
![png](output_16_1.png)
    



```python
predictions = np.argmax(model2.predict(test_images), axis=1)

cm = confusion_matrix(test_images.labels, predictions)
clr = classification_report(test_images.labels, predictions, 
                            target_names=test_images.class_indices, 
                            zero_division=0)
```

    6/6 [==============================] - 2s 351ms/step



```python
plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(num_c) + 0.5, labels=test_images.class_indices, rotation=90)
plt.yticks(ticks=np.arange(num_c) + 0.5, labels=test_images.class_indices, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```


    
![png](output_18_0.png)
    

