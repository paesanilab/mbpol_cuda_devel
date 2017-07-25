C++ code based on CUDNN/CUBLAS/CUDA, built for Neural Network model.
It will read in weights/bias from HDF5 file, create layer list and make predictions.
Supporting both single and double precision floating point.

Layer model support : 
fullyconnectforward layer
ReLU activiation layer
Hyperbolic tangential activiation layer
softmax activiation layer at present.

Two testers are provided:
1) A single precision tester, "testcase_forCUDA_single", loads hdf5 file "32_2b_nn_single.hdf5" 
and creates five of fully_connect + activiation layers and one fully_connect + linear layer.
Then it loads 11 sample inputs, and predicts for the final scores.
2)  A double precision tester, "testcase_forCUDA_double".  It will operate exactly as the above one,
except all input data are double-precision.

