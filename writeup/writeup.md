## **Traffic Sign Recognition** 

In this project we are buildung a traffic sign classifier using the German Traffic Sign Dataset. The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/histogram.png "Distribution of Train, Test, Validation Data"
[image2]: ./images/signs.png "Class Labels (Signs)"
[image3]: ./images/new_train_distribution.png "New Train Data Set Distribution"
[image4]: ./images/augmented.png "Augmentation"
[image5]: ./images/grayscale.png "Grayscaling"
[image33]: ./examples/random_noise.jpg "Random Noise"
[image43]: ./examples/placeholder.png "Traffic Sign 1"
[image53]: ./examples/placeholder.png "Traffic Sign 2"
[image63]: ./examples/placeholder.png "Traffic Sign 3"
[image73]: ./examples/placeholder.png "Traffic Sign 4"
[image83]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration 

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library and python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410. 
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the train, validation and test data are distributed. As you can see in the illustration below, the train data set ist not uniformly distributed. This could mean that less-data classes are less likely to be predicted than classes with more data. 

![alt text][image1]

To get an idea what the images in the German Traffic Signs Dataset look like, I visualize one sign of each class.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to generate additional data because of the imbalance of the train data set. 
To add more data to the the data set, I rotate, contrast and sharpen the images randomly. I use this data augmentation technique to the classes with less samples, so that the final train data set is more balanced as we can see in the following illustration.

![alt text][image3]

Here is an example of an original image and an augmented image:

![alt text][image4]

Next, I decided to convert the images to grayscale to reduce the dimensions and complexity of the neural network. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

As a last step, I normalized the image data to a range of [0,1] by using *pixel / 255*, so the network can treat every feature equally.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscale image   							             | 
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 28x28x32 	 |
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 24x24x32 	 |
| Max pooling	      	   | 2x2 stride,  outputs 12x12x32 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 8x8x64 	 |
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 4x4x64 	 |
| Max pooling	      	   | 2x2 stride,  outputs 2x2x64 				|
| Flatten     	         |	256                          |
| Fully connected		     | 128, activation: relu, keep prob: 0.5        									|
| Fully connected		     | 64, activation: relu, keep prob: 0.75        									|
| Output				            | 43, activation: softmax        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a fix learning rate 1e-3. The batch size was set to 32 images. The weights were initialized by a truncated normal distribution. The network was trained for 10 epochs on a notebook and it takes about 1 hour. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of  100%
* validation set accuracy of 98.5% 
* test set accuracy of 96.5%

If an iterative approach was chosen:
**What was the first architecture that was tried and why was it chosen?

*As first approach, I was trying the LeNet architecture from the LeNet lab on normalized data. These architecture was quite enough to reach an validation accuracy > 93%.* 

* What were some problems with the initial architecture?

*The problem of the initial architecture was that the model was not good enough and its poor performance on the test set.*   

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. 
A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

