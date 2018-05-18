# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rensandao/CarND-term1-p2-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **12630**
* The size of test set is **4410**
* The shape of a traffic sign image is **(32,32,3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed under every different sign.
As is shown, some of data is obviously higher, while some others look like too low.

![visualization][./visualized images/bar_chart.png "visuliazation"]

### Design and Test a Model Architecture

#### 1. Preprocess the image data.

For a first try, I converted the images to grayscale. Because basically color is not main feature in sign classifier, using grayscale can make it easier for Iterative calculation.

After grayscale, I applied Data normalization to get the input features data close in certain scope(-1,1). That can make Loss Optimization more concentrated and faster.

With them, the final valudation accuracy can achieve 0.899 at most. This is obviously lower than aquirement(0.93).

After watching images from the train set, I found some of them have low background contrast, which quitely make it hard to classify. So
I decided to apply CV library- [CLAHE](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) to promote contrast. And it did work. The validation accuracy comes to 94.1%, an obvious improvement. 


Here is an example of a traffic sign image before and after grayscaling.

![alt text][./new_images/1.png]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Model architecture

My model consisted of the following layers:

| Layer         		|     Description	        					                       | Input | Output |
|:---------------------:|:---------------------------------------------:|:----:|:----:| 
| 1.Convolution 3x3    | 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x6|
| RELU					            | 							activation                             |           | 						|
| Max pooling			       | 2x2 stride, 2x2 kernel size						           |28x28x6    |14x14x6|
| 2.Convolution 5x5 	  | 1x1 stride, valid padding, RELU activation 	|14x14x6    |10x10x16|
| RELU					            | 							activation                           |           | 						|
| Max pooling			       | 2x2 stride, 2x2 kernel size	   					        |10x10x16   |5x5x16 |
| Flatten				          |          flatten 3D to 1D  					            |5x5x16     | 400   |
| 3.Fully Connected    |   connect some layers                			    |400        | 120   |
| RELU					            | 							activation                           |           | 						|
| dropout				          | 		 drop some datas avoiding 	overfitting    |           | 						|
| 4.Fully Connected    |          connect some layers                 |120|84|
| RELU					            | 							activation                            |         | 						|
| 5.Fully Connected    | 	      connect all 43 layers                 |83       | **43**|
 

#### 3. Model tranning 

To train the model, I used a classic model architecture-LeNet5 for traffic signs classification, which was also knew according to Udacity class lab. Learn to use it, that is more important.

But to use it, it needed some changed. The Subsampling layers were replaced with max pooling layers, and activation function used ReLU which helps created nonlinear results. Counting the total sign classes, the output classes was thus set to 43. The optimizer was AdamOptimizer, as it was.

During adjustment, I also added a Dropout layer to see whether it help release over-fitting. It turns out to be pretty well, though the train accuracy is a (0.1-0.2) lower than before. 

The number of epochs has also been related to trainning effect. worrying it took too long, I only set it to 20 at most. Learning rate was set to 0.001 at first and then 0.0008. As epoch was low, I didn't see obvious change for now.

Here are my parameters for tranning:
* rate= 0.0008
* EPOCHS = 20
* BATCH_SIZE = 128
* SIGMA = 0.1


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.996**
* validation set accuracy of **0.941**
* test set accuracy of **0.927**

As is written above, I did some adjustments to make it suitable for traffic signs classification, based on the classic LeNet-5 architecture. The main changes of accuracy are as following:
* With the adjusted architecture LeNet5, and process data with grayscale and normalization, I got 0.984 and 0.899 respectively for trainning and validation accuracy. I changed the EPOCH from 10, 15 to 20 to see whether it got better, which turned out to negative.
* After adding the CLAHE to promote the contrast, I got 0.998 for trainning set and 0.947 for validation. I wondered it was over-fitting, so I lately adding 1-2 dropout layer with keep_probability of 0.8 , found the accuracy got slightly decreased. 


### Test the Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Actually, I think these five images can be easy to dectect, because it is clear and we can easily distinguish them with our real eye. However, it came out to be a little different.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     | Right-of-way at the next intersection						| 
| Speed limit(30km/h)     			| Speed limit(30km/h) 									|
| Turn left ahead					| Turn left ahead									|
| General caution	      		| General caution					 				|
| Speed limit(60km/h)			| Speed limit(50km/h)  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy of 92.7% on the test set. It sounds reasonable when it comes to a small set of samples. But it failed classifying the fifth image, which 
made me feel a little unhoped and less confident on my deep learning architecture. It may tell I need to improve accuracy with more data preprocess.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Firstly, the model showed absolutely certain to these five images with all probability of 100%.
The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection   									| 
| 1.0     				| Speed limit(30km/h) 										|
| 1.0					| Turn left ahead											|
| 1.0	      			| General caution						 				|
| 1.0				    | Speed limit(50km/h)     							|


For the fifth image, the model is relatively sure that this is a Speed limit (probability of 1.0), but with 50km/h. Luckily it is classified to the low speed limit. If were classified to higher speed scope, it can lead to bad consequence in real situation. Thus, more work need to do to improve it.

Next, I will consider more adjustments as following:
* Expand and fake some data with low distributed rate, as DNN tends to be partial to those with high rates like we people do.
* Train the network for higher epochs to explore the accuracy's maximum stability.
* Choose some other German traffic images to test the effect after improvement. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


