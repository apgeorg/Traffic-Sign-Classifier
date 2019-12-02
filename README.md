## **Traffic Sign Classification** 

This project was submitted for the Udacity Self-Driving Car Nanodegree program. 
In this project I built and trained a deep neural network to classify traffic signs in Tensorflow using the German Traffic Sign Dataset. The goals / steps of this project are the following:

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
[image6]: ./images/sign1.jpg "Traffic Sign 1"
[image7]: ./images/sign2.jpg "Traffic Sign 2"
[image8]: ./images/sign3.jpg "Traffic Sign 3"
[image9]: ./images/sign4.jpg "Traffic Sign 4"
[image10]: ./images/sign5.jpg "Traffic Sign 5"
[image11]: ./images/web_signs.png "Web Signs"
[image112]: ./images/top5_sign1.png "Traffic Sign 1 Prediction"
[image113]: ./images/top5_sign2.png "Traffic Sign 2 Prediction"
[image114]: ./images/top5_sign3.png "Traffic Sign 3 Prediction"
[image115]: ./images/top5_sign4.png "Traffic Sign 4 Prediction"
[image116]: ./images/top5_sign5.png "Traffic Sign 5 Prediction"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration 

I used the pandas library and python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410. 
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the train, validation and test data are distributed. As you can see in the illustration below, the train data set ist not uniformly distributed. This could mean that less-data classes are less likely to be predicted than classes with more data. 

![alt text][image1]

To get an idea what the images in the German Traffic Signs Dataset look like, I visualize one sign of each class.

![alt text][image2]

### Design and Test a Model Architecture

As a first step, I decided to generate additional data because of the imbalance of the train data set. 
To add more data to the the data set, I rotate, contrast and sharpen the images randomly. I use this data augmentation technique to the classes with less samples, so that the final train data set is more balanced as we can see in the following illustration.

![alt text][image3]

Here is an example of an original image and an augmented image:

![alt text][image4]

Next, I decided to convert the images to grayscale to reduce the dimensions and complexity of the neural network. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

As a last step, I normalized the image data to a range of [0,1] by using *pixel / 255*, so the network can treat every feature equally.

#### Model Architecture

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

#### Training

To train the model, I used an Adam optimizer with a fix learning rate 1e-3. The batch size was set to 32 images. The weights were initialized by a truncated normal distribution. The network was trained for 10 epochs on a notebook and it takes about 1 hour. 

#### Model Results

My final model results were:
* training set accuracy of  100%
* validation set accuracy of 98.5% 
* test set accuracy of 96.5%

If an iterative approach was chosen:

##### What was the first architecture that was tried and why was it chosen?

*As first approach, I was trying the LeNet architecture from the LeNet lab on normalized data. These architecture was quite enough to reach an validation accuracy > 93%.* 

##### What were some problems with the initial architecture?

*The problem of the initial architecture was that the model was not good enough and its poor performance on the test set.*   

##### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

*First, I added dropout layers between the fully connected layers to prevent overfitting and then I added more convolutional  layers before performing pooling as the well-known VGG16 architecture do.*

##### Which parameters were tuned? How were they adjusted and why?

*It was more a try and error process. I was adjusting kernel and filter size of the convolution layers to get more features etc. Furthermore I was adjusting the keep probability rate if the validation accuracy was much more less than the training accuracy. Learning rate and epochs were not adjusted because they were suitable enough. A good practice to tune parameters is to perform grid search.*  

##### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

*An important design choice is to use dropout layers, which randomly ignores neurons during training. The effect is that the network becomes less sensitive to the specific weights. That means that the network is more generalized and is less likely to overfit the training data. In general, convolutional layers are quite good for image classification because they are extracting features of images. In addition, I think that double the filter size of a convolution layer after performing max pooling is good practice.*

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image11]

All images show good brightness conditions and look very ideal for the classifier. Possible difficulties could be caused by the copyright labels in the pictures. 

#### Model Predictions

Here are the results of the prediction:

| Image			            |     Prediction	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road  									              | 
| Roundabout mandatory  | Roundabout mandatory  										    |
| Speed limit (30km/h)	| Speed limit (30km/h)											    |
| Keep right	      		| Keep right					 				                  |
| Stop sign		          | End of no passing by vehicles over 3.5 metric tons |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is not comparable with  the accuracy on the test set because the test set contains more samples. 

For the first image, the model is 100% sure that this is a priority road sign (probability of 100%). The other top five softmax probabilities are zero. 

![alt text][image112]

For the second image, the model is relatively sure that this is a roundabout mandatory. It achieves a probability of 55% for the "Roundabout mandatory" sign. The other top five softmax probabilities are shown in the illustration below.

![alt text][image113]

For the third image, the model is sure that this is a speed limit (30km/h) sign	with a probability of 100%. 

![alt text][image114]

For the fourth image, the model is again sure that this is a keep right sign	with a probability of 100%.  

![alt text][image115]

For the fifth image, the model misclassified the stop sign. In other words, the model predicts an end of no passing by vehicles over 3.5 metric tons sign with a probability of 60%. Furter, the stop sign is not listed in the top five softmax probabilities which means that the model has trouble predicting on stop signs.

![alt text][image116]

