# Implementing Neural Network model based on CUDA/CUDNN/CUBLAS, and two tester cases of "NN_L2H2O_poly2d" with different precisions

## General
These codes model the complete workflow of performing prediction based on Neural Network model in C++ and with CUDA/CUDNN/CUBLAS libraries and HDF5 support.  
It will read out layers names, weights and bias from an HDF5 file in sequence, then create layers based on their type, and predict final scores by the input samples.

## File list
- *error_util.hpp*                 : File for handling exception info of cuda/cublas/cudnn
- *whichtype.hpp*                  : File for testing the data type in a template
- *readhdf5.hpp*                   : Class to read in dataset saved in a HDF5 file. Need HDF5 library support.
- *network.cu*                     : Class to model neural network layer/algorithm. Need CUDA/CUDNN/CUBLAS supports.
- *NN_L2H2O_poly2d.cu*             : Tester main function, with both single/double floating point precisions
- *NN_L2H2O_poly2d.in*             : Input samples data
- *32_2b_nn_single.hdf5*           : HDF5 file for the single floating point test
- *32_2b_nn_double.hdf5*           : HDF5 file for the double floating point test
- *python.out*                     : Reference Output result from Python Keras/Theano
<br>

### For class file *readhdf5.hpp* reading HDF5 file:  
This file uses HDF5 library C++ API, and offers functions:
   - to read an attribute name and saved data (usually strings) in a HDF5 file group, and retain the order.
      - This function can be used to retrieve Neural Network layer names in the group "/model_weights/" of a given HDF5 file
      - Or can be used to retrieve dataset names (weight and bias) saved for each layer
   - to read out a dataset data by the dataset path in the HDF5 file.
      - This function can be used to obtain a layer's weight and bias, given the absolute path is retrieved by the first function. 
    
### For class file *network.cu* modelling Neural Network layers and algorithms:
This file uses CUDA/CUDNN/CUBLAS libraries, and offers:  
   - Layer creation, including a normal dense layer and an activiation layer. Saved weights and bias are initialized on both host and device.
   - Neural Network algorithms, include:
       - fully connected forward, based on function `cublasSgemm()` or `cublasDgemm()` according to the precision of data
          - `cublasSgemm()` is a function utilizing cuda library to perform fast algebra dot product of *m[atrix]* and *m[atrix]* in **single** floating point precision. And similarly, 
          - `cublasDgemm()` perform *matrix* dot *matrix* in **double** float precision
          - `cublasDgemv()` perform *matrix* dot *vector* in **double** float precision
       - softmax forwards, using `cudnnSoftmaxForward()`
       - Activation_TANH forwards, using `cudnnActivationForward()` with CUDNN_ACTIVATION_TANH to define the activiation type as hyperbolic tangential
       - Activation_ReLU forwards, using `cudnnActivationForward()` with CUDNN_ACTIVATION_RELU to define the activiation type as ReLU nonlinearity
   - Layer list creation, saving a list of layers and automatically performing prediction according to layer types. 

### For the provided tester:  
These testers are from repo /paesanilab/NeuralNets/testcase_forCUDA/ which are originally written in Python with Keras/Theano support.  
In the repo, the trained layers weights/bias are saved in a HDF5 file, and testing samples are written to an text file (both in different precison, respectively).  
The above files are loaded by this C++ code to reproduce similar results as in Python, for the purpose of testing CUDA/CUDNN/CUBLAS utility in C++.  
File *python.out* contains the results from Keras/Theano based Python program.  

The results from single precision tester have an difference of less than 1e-7, and the difference by double precision is at least less than 1e-15.
    
## TO RUN
To make executive files:
   - Make sure cuda/cudnn and hdf5 libraries are installed. Makefile will look for environment variable *CUDA_PATH/CUDNN_PATH/HDF5_PATH* to locate the installed library and included header files. If not found, it will look for /usr/local/cuda and /usr/local/hdf5 If not found, it will fail.
   - If the libraries path are correctly found, type "make" to make two executive files.
   - Run `NN_L2H2O_poly2d` for the tester.
   - Compare the final output scores with what are from Python Keras/Theano.



