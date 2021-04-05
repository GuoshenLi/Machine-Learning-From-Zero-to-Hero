
# Machine-Learning-From-Zero-to-Hero
Some example codes of widely used machine learning algorithms

## Linear Regression
The original data looks like this:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/Linear_Regression/data1.png width = '642' height = '342'/><br/>

After running gradient descent or normal equation:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/Linear_Regression/linear_reg1.png width = '642' height = '342'/><br/>

The contour of the cost function:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/Linear_Regression/contour_cost.png width = '642' height = '342'/><br/>

## Support Vector Machine (Using SMO Algo to Optimize)

### Linear Kernel
The original data looks like this:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/svm/data_linear.png width = '642' height = '342'/><br/>

After training the SVM with a linear kernel, we can get the following decision boundary:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/svm/svm_linear.png width = '642' height = '342'/><br/>

### Gaussian Kernel

The original data looks like this:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/svm/data_gaussian.png width = '642' height = '342'/><br/>

After training the SVM with a Gaussian kernel, we can get the following decision boundary:


<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/svm/svm_gaussian.png width = '642' height = '342'/><br/>



## K-Means

### 2D clustering

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/kmeans/three_class.png width = '642' height = '342'/><br/>

### Image Compression (From Original 0 ~ 255 (8bit) to 0 ~ 16 (4bit))

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/kmeans/image_compression.png width = '642' height = '342'/><br/>



## Principle Component Analysis

### 2D-1D

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/pca/2d_1d_line.png width = '642' height = '642'/><br/>

### EigenFaces

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/pca/original_faces.png width = '642' height = '642'/><br/>

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/pca/recovered_faces.png width = '642' height = '642'/><br/>



## Gaussian Mixture Model

### Visualize the clusters

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/GMM/GMM_Contour.png width = '642' height = '342'/><br/>

### The loglikelihood

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/GMM/GMM_likelihood.png width = '642' height = '442'/><br/>


## Neural Network

### Visualize the decision boundary

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/neural_network/Neural_Network_Decision_Boundary.png width = '642' height = '342'/><br/>

### The loss and training acc

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/neural_network/cost_function.png width = '742' height = '442'/><br/>


<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/neural_network/train_acc.png width = '742' height = '442'/><br/>

# MCMC sampling


<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/MCMC/mcmc.png width = '695' height = '516'/><br/>

# Bayesian Linear Regression

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/bayesian_regression/bayesian_reg.png width = '695' height = '516'/><br/>


# (Fully Convolutional Neural Network) U-Net-Based-Heart-Sound-Segmentation

This is the project that I done in the lab during my undergraduate study.

A given heart sound can be segment into four stages which is shown as below:

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/U-Net-Heart-Sound/1.png width = '542' height = '280'/><br/>

Our goal is to train a deep neural network to segment a given heart sound into 4 stages.

The architecture that we use is 1-D U-Net used in the literature.

<img src = https://github.com/GuoshenLi/Machine-Learning-From-Zero-to-Hero/blob/main/U-Net-Heart-Sound/2.png width = '742' height = '280'/><br/>

