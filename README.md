# Dog Breed Classifier with CNNs
## Introduction
Dog breed classification is a rather tricky classification problem because there are so many different breeds of dogs and that visually a lot of them look very similar, and some same breed dogs have different color. Thus, for the model to classify image of a dog properly there must be a robust algorithm to address this problem. For this project, Convolutional Neural Networks (CNNs) were utilized because it proves to be effective for image classification problems. CNNs were used in two ways, firstly a model was built and trained from scratch and then a model using transfer learning was utilized to maximize accuracy. In addition to classifying dog images based on the breed of the dog, the algorithm can also take as input an image of a human and classify it into any resembling dog breed. There are images of dogs of 133 different breeds in the dataset used for this project, although there are a lot more than that.

## Problem Statement
In simple words the problem statement is, When given input image of a dog, identifying the breed of that dog and when given image of a human, identifying the resembling dog breed.
The classifier solves two problems
Dog Breed Classification: The classifier will be able to classify images of dogs into different breeds.
Human face classification: The classifier will classify human face image into resembling breed of dog.


##  Datasets used
Two datasets were used during this project
[Dog Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip): Dog Dataset containing images of different dog breeds. It contains 8351 images having 131 different dog breeds. Images are divided into train and test datasets
 [Human Dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz): This dataset includes a set of human images to be used during the project. It contains 13233 images.

## Tools and Libraries Used
- Python
- Jupyter Notebooks
- Numpy
- Glob
- Open CV and haarcascades
- Matplotlib
- tqdm
- torch and torchvision
- PIL
- os

Note: GPU based machine was used for training and processing of this project. 

## References and Supporting Material
1.	**Base Repository**: [Original GitHub repository](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/) used as a base for this project. 
2.	**Domain Knowledge**: [Using Convolutional Neural Networks to classify dog breeds](http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf)
3.	**Datasets**: [Dog Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [Human Dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
4.	**Benchmark Models**: [Deep Learning and Transfer Learning approaches for Image Classification](https://www.researchgate.net/publication/333666150_Deep_Learning_and_Transfer_Learning_Approaches_for_Image_Classification)
