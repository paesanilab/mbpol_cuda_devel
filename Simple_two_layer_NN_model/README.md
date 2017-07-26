Implementing a simple two_layer Neural Network

This is an example showing the basic workflow and utilization of CUDNN to solve Neural Network problems.

This example is an exercise from open course CS231n Convolutional Neural Networks for Visual Recognition from Stanford university: http://cs231n.stanford.edu/
The credit will be given to the authors of the course material.
In this work we will develop a neural network with fully-connected layers to perform classification.
The NN model consists of two layers: 
     1) A fully connected layer, with ReLN nonlineary. Or in mathmatical expression:
          h1=max(0,x∗W1+b1)
     2) A fully connected layer:
          h2=h1∗W2+b2
and 
     3) The loss function is the softmax loss :
          loss= - log( exp(h2_y) / ∑exp(h2_j) )
See Python file for more explanation.


The example is given in two versions:
1) A Python script, which was the originally designed from the course, showing the clear walk path of the basic concept of NN.

2) A C++ code based on CUDNN/CUBLAS/CUDA, showing how to use CUDNN to solve the NN problem.
     It uses the "error_util.h" and "gemv.h" file coming from the "mnistCUDNN" example composed by nVIDIA.
     The core of the code is class file "network.cu", in which the current implemented definitions include:
          - Layers, its input/output dimension (m,n), the weighted matrix (W[m,n]), a bias vector (b[n])
          - Algorithm between layers, including:
               - simple fully connected forward, using `cublasSgemv()` function
                         - `cublasSgemv()` is a function utilizing cuda library to perform fast algebra dot product 
                           of 'm[atrix]' and 'v[ector]' in single floating point precision.
                         - Similar functions include :
                              - `cublasDgemv()` : perform matrix dot vector in double float precision
                              - `cublasSgemm()` : perform matrix dot matrix in single float precision
               - softmax forwards, using `cudnnSoftmaxForward()`
               - Activation_TANH forwards, using `cudnnActivationForward()` with CUDNN_ACTIVATION_TANH
               - Activation_ReLU forwards, using `cudnnActivationForward()` with CUDNN_ACTIVATION_RELU
     File "two_layer_NN.cu" is the tester, in which layers are created and predictions are made according to the above two layers algorithms.
     This code works only with single precision float at present.

In the offered example, the Python tester generates a random two layers model, with 4x10 dims in the first layer and 10x3 in the second. 
Then, it creates 5 random samples (each in a vector of size 4), and is given the correct classifiers. 
After this step the Python script predicts the scores after all layers, and calculates the losses of all samples according to the classifiers.

In the CUDA/CUDNN tester, the weights/bias/input_samples/classifiers generated randomly by the Python code are copied/pasted into an input file, 
and are input to the CUDA/CUDNN tester as arraies when the file is loaded. 
Then the scores and losses of all samples are predicted (in sequence) and compared with what Python gives out.

The results from these two testers show great consistency, with around  1e-7 difference by single float precision.


     
     
     
To run:
1) For Python, open "two_layer_net.ipynb" in Jupyter Notebook and run
2) For C++, make sure CUDNN/CUBLAS/CUDA are installed. The compile will search for "CUDA_PATH" and "CUDNN_PATH" for the installed libraries.
     Run "make build" to compile "two_layer_NN"
     Usage: ./two_layer_NN  [-device=0]   to run the test case on selected device
     
     OR, "make" to clean/build/run the file all at once on default device.
