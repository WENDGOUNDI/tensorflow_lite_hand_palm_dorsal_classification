# Tensorflow Lite Hand Palm Dorsal Classification
Why should we bother recognize hand position, palmar or dorsal? Well with the fast growing of technology, machines and humans are working closely. Machines are assigned difficult tasks to support humans in their work, mostly in factories. Gesture recognition is a hot computer vision topic used in robotics. Gesture recognition has a wide range of potential, such as human-computer interaction, control a robot or approve or disapproved a scenario. For this project, we trained an AI model that is able to identify hand position: palmar or dorsal. We also converted the TensorFlow model to a TensorFlow Lite model **to reduce the model size and inference time.**

# Dependencies
```
Zipfile
OS
Tensorflow
Tensorflow Lite
Pathlib
Numpy
Livelossplot
Matplotlib
```

# Dataset
For this project, we have used the public **Hands and palm images dataset** from kaggle. The dataset is large of 11076 images with 2 classes. Each image is an RGB image of size 1600 x 1200 pixels. The images have been collected on 190 subjects from different skin color, aged from 18 to 75 years old. During the data collection, The participants were instructed to alternate between opening and closing the fingers on both their right and left hands. Afterwards, photographs were taken of each hand from both the dorsal and palmar sides against a white background, ensuring that they were positioned at a consistent distance from the camera.

Link: https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset

# Training
The dataset has been split into 3 parts: training set 7532 images, validation set 1328 images, testing set 2216 images. The model is based on a VGG16 pretrained layers with a customize fully connected layers for classification. The input shape equal to (224, 224, 3) corresponding to VGG16 default input shape. Data augmentation has been applied to increase the data size as well as to help the model to generalize well. `categorical_crossentropy` has been used as training loss and `adam` as optimizer with a learning rate set at **0.001**. Batch size of 64 for a training duration of 20 epochs. tensorflow.keras `ModelCheckpoint` function is used to only save the best weights and `EarlyStopping` on `val_loss` is applied to stop training once the model is not improving.
`Livelossplot` allows a realtime time training and validation accuracy and loss plot.

# Training Results:
![Capture](https://user-images.githubusercontent.com/48753146/235815815-44a9b985-373d-4eb1-8f1e-5fccbe73ea28.PNG)
Testing accuracy is 99.05%

# Confusion Matrix
![download](https://user-images.githubusercontent.com/48753146/236080885-de40c973-d5c4-4462-b585-53355d589df6.png)

# Model Optimization
While large deep learning models can be very effective for certain tasks, there are some issues that can arise during inference, which is the process of using a trained model to make predictions or classifications on new data. Some examples of issues that can arise during inference with large models are latency and performance, memory requirements, energy consumption, scalaility. To address this issue, we converted our TensorFlow model to a TensorFlow Lite model. This is a first step toward optimization for running deep learning models on edge devices or any inference device when with have limited resources. Despite we are inferencing our model in our laptop, we converted the model to TFLite model for fast inferencing.

 - TF model size: 1282.559 mb
 - TFLite model size: 464.919 mb

TFLite model size is half the TF model size. Prediction time is half as well.

![model](https://user-images.githubusercontent.com/48753146/236083083-73cc6ad2-883d-4214-9e47-02f7d302db8e.PNG)

# Prediction
### Example 1
![prediction](https://user-images.githubusercontent.com/48753146/236092218-20dc69bb-58c2-48ce-b82a-7ba79f9105bc.PNG)

### Example 2
![test2](https://user-images.githubusercontent.com/48753146/236096382-c43b1f4c-89b8-43ad-b145-822314caadf5.PNG)


