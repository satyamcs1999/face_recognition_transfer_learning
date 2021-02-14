__<h1>Face Recognition using Transfer Learning</h1>__

<p>First of all, let’s all understand what <b>Transfer Learning</b> is , basically it’s a method developed for a task which is reused as a starting point for a model on a second task.</p><br>

<p>Advantages of Transfer Learning includes :</p>

<ol>
  <li><b>Less training data :</b> Starting to train a model from scratch is a lot of work and requires a lot of data. For example, if we want to create a new algorithm that can detect a frown, we need a lot of training data. Our model will first need to learn how to detect faces, and only then can it learn how to detect expressions, such as frowns. Instead, if we use a model that has already learned how to detect faces, and retrain this model to detect frowns, we can accomplish the same result using far less data.</li><br>
  <li><b>Models generalize better :</b> Using transfer learning on a model prepares the model to perform well with data it was not trained on. This is known as generalizing. Models that were trained using transfer learning are better able to generalize from one task to another because they were trained to learn to identify features that can be applied to new contexts.</li><br>
  <li><b>Makes deep learning more accessible :</b> Working with transfer learning makes it easier to use deep learning. It’s possible to obtain the desired results without being an expert in deep learning, by using a model that was created by a deep learning specialist and applying it to a new problem.</li><br>
</ol>

<p>For Facial Recognition using Transfer Learning, <b>MobileNet</b> architecture is being used and the implementation is using <b>Keras</b> Library</p><br>

<div align="center">

![Face_Recognition](https://miro.medium.com/max/361/1*Vf2k5KEQi6fGD4sGSW1ixg.jpeg)

</div><br>

<h2> Loading and Setup of MobileNet </h2>

```python
from keras.applications import MobileNet

img_rows, img_cols = 224, 224 

MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

for layer in MobileNet.layers:
    layer.trainable = False
```

<br>
<p>In this code snippet, we first import MobileNet module from Keras ,after which we set the no. of pixels in image row and image column as 224 as MobileNet is designed to work on 224 x 224 pixel input images size</p><br>
<p>Furthermore, we reload the MobileNet model without top or <b>Fully Connected Layers</b></p><br>
<p>We freeze all the layer of MobileNet layer by setting layer.trainable as <b>False</b>, so that it doesn’t get re-trained when we train our Face Recognition Dataset.</p><br>

<h2> Creation of Fully Connected Layer </h2>

```python
def topLayer(bottom_model, num_classes):
   top_model = bottom_model.output
   top_model = GlobalAveragePooling2D()(top_model)
   top_model = Dense(1024,activation='relu')(top_model)
   top_model = Dense(1024,activation='relu')(top_model)
   top_model = Dense(512,activation='relu')(top_model)
   top_model = Dense(num_classes,activation='softmax')(top_model)
   return top_model
```

<p>In the above code snippet , we create a <b>topLayer</b> method,whose parameters are <b>bottom model</b> and <b>no.of classes</b> , it creates the top of the model that would be placed on top of the bottom layers.</p><br>

<h2> Addition of Fully Connected Layer to MobileNet </h2>

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = topLayer(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())
```

<p>The primary function of the code snippet above is to attach the top of the model created using <b>topLayer</b> method and attach it to MobileNet.</p><br>
<p>First , we import necessary Keras models and layers and declare the no. of classes as 2 (as we are classifying 2 Faces) and then call <b>topLayer</b> method and pass MobileNet layers and no. of classes arguments to it and store the generated value to a variable named <b>FC_Head</b></p><br>
<p>We create our model by passing <b>MobileNet’s input</b> as input and <b>FC_Head</b> as output, the summary of model created is as follows:</p><br>

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
conv1_pad (ZeroPadding2D)    (None, 225, 225, 3)       0         
_________________________________________________________________
conv1 (Conv2D)               (None, 112, 112, 32)      864       
_________________________________________________________________
conv1_bn (BatchNormalization (None, 112, 112, 32)      128       
_________________________________________________________________
conv1_relu (ReLU)            (None, 112, 112, 32)      0         
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 112, 112, 32)      288       
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 112, 112, 32)      128       
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 112, 112, 32)      0         
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 112, 112, 64)      2048      
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 112, 112, 64)      256       
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 112, 112, 64)      0         
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 113, 113, 64)      0         
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 56, 56, 64)        576       
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 56, 56, 64)        256       
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 56, 56, 64)        0         
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 56, 56, 128)       8192      
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 56, 56, 128)       1152      
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 56, 56, 128)       16384     
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 57, 57, 128)       0         
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 28, 28, 128)       1152      
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 28, 28, 128)       512       
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 28, 28, 128)       0         
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 28, 28, 256)       32768     
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 28, 28, 256)       2304      
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 28, 28, 256)       65536     
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 29, 29, 256)       0         
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 14, 14, 256)       2304      
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 14, 14, 256)       1024      
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 14, 14, 256)       0         
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 14, 14, 512)       131072    
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 15, 15, 512)       0         
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 7, 7, 512)         4608      
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 7, 7, 512)         2048      
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 7, 7, 512)         0         
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 7, 7, 1024)        524288    
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 7, 7, 1024)        9216      
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 1026      
```

<br>
<h2> Loading of Image Dataset </h2>

```python
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'Dataset/train/'
validation_data_dir = 'Dataset/validation/'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 9

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
```

<p>We first load <b>ImageDataGenerator</b> module from <b>Keras</b></p><br>
<p>In the above code snippet , we load our Face Dataset that consist of images of two faces , also , it is further subdivided into <b>Training Images</b> and <b>Validation Images</b>.</p><br>
<p>Furthermore, we perform Data Augmentation on Training as well as Validation Images i.e., <i>train_datagen</i> and <i>validation_datagen</i> ,and the batch size declared is 9. After which , we create a generator function for both Training and Validation i.e., <i>train_generator</i> and <i>validation_generator</i>.</p><br>

<h2> Model Training </h2>

```python
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("face_recognition.h5", monitor="val_loss",
                             mode="min", save_best_only=True,
                             verbose=1)
earlystop = EarlyStopping(monitor = "val_loss",
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights = True)
callbacks = [earlystop, checkpoint]

model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

nb_train_samples = 800
nb_validation_samples = 196
epochs = 3
batch_size = 9

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
```

<p>In the above code snippet, we import <b>RMSprop Optimizer</b> from Keras , also we import <b>ModelCheckpoint</b> and <b>EarlyStopping</b> from Keras.<b>Callback</b>.</p><br>
<p><b>ModelCheckpoint</b> is basically used to save model or weight at an interval whereas in case of <b>EarlyStopping</b> , its function is to stop training as soon as monitored metric i.e. “val_loss” in this case ,stops improving. Also the batch size is 9.</p><br>
<p>The number of <b>epochs</b> is set to 3 and <b>fit_generator</b> is used for training the model for specified number of epochs.</p><br>

![Epochs](https://miro.medium.com/max/875/1*vUPKFmg7HH3SIgcqnyKwWA.png)<br>

<p align="center"><b>Epochs</b></p><br>

<h2> Loading and Testing of Classifier Model </h2>

```python
from keras.models import load_model

classifier = load_model('face_recognition.h5')
```

<p>The above code snippet loads the saved model “<b>face_recognition.h5</b>” to the variable “<b>classifier</b>” using Keras module.</p><br>

```python
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_recognition_dict = {"[0]": "Arnav ", 
                      "[1]": "Satyam"}

face_recognition_dict_n = {"Arnav": "Arnav ", 
                      "Satyam": "Satyam"}

def draw_test(name, pred, im):
    f = face_recognition_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, f, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + face_recognition_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,20):
    input_im = getRandomImage("Dataset/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()
```

<p>In the above code snippet , we first import necessary modules i.e. <b>os</b>, <b>cv2</b> and <b>numpy</b> and their required sub-modules.</p><br>
<p>After which, we create a dictionary for specifying index corresponding to images contained in the folder specified. We also create another dictionary for specifying label corresponding to images contained in the folder (of same name) specified.</p><br>
<p>The “<b>draw_test</b>” method is used for setting up the texture of predicted image output i.e. image’s background , image’s caption’s font and many more.</p><br>
<p>The “<b>getRandomImage</b>” method loads images from a random folder in our test path.</p><br>
<p>Using range method in for loop, we could specify the number of images to be tested by trained classifier for validation purpose.</p><br>

<h2> Output </h2>

![First_Output](https://miro.medium.com/max/875/1*4DWL-bQ6v2h0IzTOPYN2-w.png)<br>

<p align="center"><b>Prediction of First Output</b></p><br>

![Second_Output](https://miro.medium.com/max/875/1*Q_BSahhbH8ESP4MECfIBww.png)<br>

<p align="center"><b>Prediction of Second Output</b></p><br>

<h2>Thank You :smiley:<h2>
<h3>LinkedIn Profile</h3>
https://www.linkedin.com/in/satyam-singh-95a266182
