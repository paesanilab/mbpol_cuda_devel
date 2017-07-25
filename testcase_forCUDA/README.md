C++ code based on CUDNN/CUBLAS/CUDA, built for Neural Network model.
It will read in weights/bias from HDF5 file, create layer list and make predictions.
Supporting both single and double precision floating point.

Layer model support : 
fullyconnectforward layer
ReLU activiation layer
Hyperbolic tangential activiation layer
softmax activiation layer at present.

File list:
- error_util.hpp    File for handling exception info of cuda/cublas/cudnn
- whichtype.hpp     File for testing the data type used for a template
- readhdf5.hpp      Class to read in dataset saved in a HDF5 file. Need HDF5 library support.
- network.cu        Class to model neural network layer/algorithm. Need CUDA/CUDNN/CUBLAS supports.

-testcase_forCUDA_single.cu   A single-float precision tester.
-testcase_forCUDA_single.in   Input samples for the above tester.
-32_2b_nn_single.hdf5         HDF5 file for the above tester

-testcase_forCUDA_double.cu   A double-float precision tester.
-testcase_forCUDA_double.in   Input samples for the above tester.
-32_2b_nn_double.hdf5         HDF5 file for the above tester


For the provided two tester:
1) A single precision tester, "testcase_forCUDA_single", loads hdf5 file "32_2b_nn_single.hdf5" 
and creates five of fully_connect + activiation layers and one fully_connect + linear layer.
Then it loads 11 sample inputs, and predicts for the final scores.
2)  A double precision tester, "testcase_forCUDA_double".  It will operate exactly as the above one,
except all input data are double-precision.

To make executive files:
1) Make sure cuda/cudnn and hdf5 libraries are installed.
Makefile will look for environment variable CUDA_PATH/CUDNN_PATH/HDF5_PATH to locate the installed library and included header files.
If not found, it will look for /usr/local/cuda and /usr/local/hdf5 
If not found, it will fail.

2) If the libraries path are correctly found,
type "make" to make two executive files.

3) Run "testcase_forCUDA_single" to test single precision.
Run "testcase_forCUDA_double" to test double precision.




