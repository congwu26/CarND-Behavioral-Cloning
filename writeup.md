#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/CNN_architecture.png "CNN Architecture"
[image2]: ./examples/left.jpg "Left camera image"
[image3]: ./examples/center.jpg "Center camera image"
[image4]: ./examples/right.jpg "Right camera image"
[image5]: ./examples/Training_process.png "Training process"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* 'model.py' containing the script to create and train the model
* 'drive.py' for driving the car in autonomous mode
* 'model.h5' containing a trained convolution neural network 
* 'model.json' containing the architecture of the model
* 'writeup_report.md' summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track 1 by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5 convolutional layers with filter sizes ranging from '3x3' to '5x5', followed by four fully connected layers. 

The model includes 'RELU' layers to introduce nonlinearity in 'model.py' by
```sh
model.add(Activation('relu'))
```

The data is normalized in the model using a Keras lambda layer (code line 34). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting in 'model.py' by 
```sh
model.add(Dropout(.5))
```

The model was trained and validated on different data sets in 'model.py' to ensure that the model was not overfitting by
```sh
train_test_split()
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track 1 with full speed 30.

####3. Model parameter tuning

The model used an adam optimizer, and changed the default learning rate to 0.0001 by
```sh
Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
```

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the standard image data provided by Udacity, which is a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used the well known network architecture built by Nvidia, you can see more details in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) link.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track with big curvature, to improve the driving behavior in these cases, I changed the angle difference between center images and side images from 0.2 to 0.3 to solve this understeering problem.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 32-77) consisted of a convolution neural network with 9 layers, including 5 convolutional layers and 4 fully connected layers:
* Input layer: 64X64X3 normalized images
* Convolution2D_1: 24 filters with (5,5) dimensions, strides = (2,2), 'same' padding
* Max Pooling: size = (2,2), strides = (1,1)
* Convolution2D_2: 36 filters with (5,5) dimensions, strides = (2,2), 'same' padding
* Max Pooling: size = (2,2), strides = (1,1)
* Convolution2D_3: 48 filters with (5,5) dimensions, strides = (2,2), 'same' padding
* Max Pooling: size = (2,2), strides = (1,1)
* Convolution2D_4: 64 filters with (3,3) dimensions, strides = (1,1), 'same' padding
* Max Pooling: size = (2,2), strides = (1,1)
* Convolution2D_5: 64 filters with (3,3) dimensions, strides = (1,1), 'same' padding
* Max Pooling: size = (2,2), strides = (1,1)
* Flattening Layer
* Dense_1: 1164
* Dense_2: 100
* Dense_3: 50
* Dense_4: 10
* Output dense layer: 1

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. But after checking the data, I noticed that most of the images are associated to 0 steering angle and the angle value of non-zero images is discrete number from -1 to 1.

Compared to the data provided by Udacity, the image sequences I recorded are not enough to cover all scenarios. The standard the data should be enough to finish training a runnable model after data augmentation process.

So I adopted the dataset provided by Udacity with 8036 scenarios and put more effort on feeding good data into training network. Following parts describe how I analysis the data and process them for a better network performance.

##### Data overview

The data contains 24,108 images, with 8,036 taken from the center, left, and right cameras each. The images in dataset are '.jpg' format with '160 X 320 X 3' dimensions. Below are one scenario sample with left, center and right images:

![alt text][image2]
![alt text][image3]
![alt text][image4]

##### Image cropping and resizing

To make the training process focusing more on the road, the images were cropped 20% from the top and bottom with new dimensions '96 X 320 X 3'. 

After this cropping process, I resized the image to '64 X 64 X 3' to reduce the input feature number of our network.

Note: this pre-processing method needs to be done in both 'model.py' and 'drive.py' to ensure the input data format of the network is same.

##### Data augmentation

Since the scenarios provided by standard data folder is limited to 8036, and most of them are associated with o steering angle, which we don't want to put into our training data, we need to augument our limited data for an more adaptive and effictive model. 

1. Flipping images: This is a simple technique to double our data by flipping the image and nagating the corresponded steering angle.

2. Brightness changes: By randomly changing the brightness of the image, we can also cover the shadow case making our data model more adaptive.

3. Use side camera images: Even the input source in 'drive.py' is the center camera, but we can also use the left and right images by adding or subtracting 0.3.

##### Training process
I finally randomly shuffled the dataset and put 10% of the data into a validation set. The generator is employed to randomly generate training batches. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The batch size was setted to 512, the sample number per epoch was setted to '512 X 50 = 25600', and the ideal number of epochs was 12 as evidenced by the validation loss not improving anymore as shown in following figure:

![alt text][image5]

####4. Test in simulator

Finally I tested my model in the simulator provided by Udacity and set the speed to 30 to make it more challenging, [here](https://youtu.be/GcWgBHoupck) is a youtube video showing my model performance.
