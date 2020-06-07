# Introduction

The goal of this exercise was to implement a model that would allow classification of clothes photographs. Image classification is one of the most fundamental problems in ML. Given a set of images that are all labeled with a single category (digits 0...9), model should predict these categories for a novel set of test images and measure the accuracy of the predictions. There are a variety of challenges associated with this task, e.g. different points of view, image deformation, etc.


There are many different techniques and models to solve the problem of image classification.

In this project, I will discuss different classification models and compare them. I will begin with simple classic machine learning algorithms: K-nearest neighbors. For a more advanced model, I will implement a Convolutional Neural Network.

It will show that KNN achieves worse classification accuracy than CNN. The implemented approaches are evaluated by the accuracy of the predictions on test photos.

# Methods

**K Nearest Neighbours (KNN)**

KNN classifier can work directly on images without feature extraction. Therefore, I didn&#39;t use extract features in any way. This technique can be described as discriminative modeling. It&#39;s purpose is to model conditional probability distribution. K-Nearest Neighbors is a non-parametric classification algorithm. The basic idea behind it is simple. Given a image to classify, find k images in the train set that are &quot;closest&quot;, that is the most similar to the test image. Assign the most frequent among k labels corresponding to neighbours to the test vector or image. I chose the same parameter k (k = 5) that was used in the Benchmark. Above this number I didn&#39;t get a better result.

To calculate the distance between two objects is used some closeness metric.

I used the optimized Euclidean distance method, also called L2 norm, without any loops:

![](file:///C:\Users\asia3\OneDrive\Pulpit\S%20T%20U%20D%20I%20A\MSiD%20laby\lab4\screenshots\1.png)

In order to compute the distance matrix efficiently I needed to vectorize the operation. By vectorizing I mean expressing the operations done on each pair of elements of matrices as an operation done on whole matrices.

![](../../../OneDrive/Pulpit/S%20T%20U%20D%20I%20A/MSiD%20laby/lab4/screenshots/2.png)

The final average accuracy of the model is **85.77%.** According to the Benchmarks result, the same classifier can achieve the result of **86.0%** accuracy. This small difference may have been caused by differences in implementation, e.g. implementation of different closeness metric or which class the algorithm has chosen if both had had the same probability.
