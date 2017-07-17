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
     It uses the "error_util.h" file coming from the "mnistCUDNN" example composed by nVIDIA.
     Currtent implemented contents include:
          - Layers, its input/output dimension (m,n), the weighted matrix (W[m,n]), a bias vector (b[n])
          - simple fully connecte forward, using cublasSgemv() function
          - softmax forwards, using cudnnSoftmaxForward()
          - Activation_TANH forwards, using cudnnActivationForward() with CUDNN_ACTIVATION_TANH
          - Activation_ReLU forwards, using cudnnActivationForward() with CUDNN_ACTIVATION_RELU
     The code works with single precision float at present.
     
     
To run:
1) For Python, open "two_layer_net.ipynb" in Jupyter Notebook and run
2) For C++, make sure CUDNN/CUBLAS/CUDA are installed. The compile will search for "CUDA_PATH" and "CUDNN_PATH" for the installed libraries.
     Run "make build" to compile "two_layer_NN"
     Usage: ./two_layer_NN  [-device=0]   to run the test case on selected device
     
     OR, "make" to clean/build/run the file all at once on default device.
