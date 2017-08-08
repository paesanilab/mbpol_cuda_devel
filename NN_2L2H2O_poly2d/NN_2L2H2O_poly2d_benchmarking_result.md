# Benchmarking result of NN_2L2OH2O_poly2d 

### This file contains :  

#### 1. the benchmarking result of `NN_2L2OH2O_poly2d` in Neural Network model on 42105 samples. This benchmarking is made to the part of the code *predicting the final score* of all samples, excluding *loading parameters, initializing layers and GPU resource context, etc*. The result records the run time on different machine/GPU, supporting library version, driver version, compiler version etc. 
#### 2. the benchmarking result of calculating potential energy `E2poly` in polynomial method. It is made on the part of the code *calculating the potential energy E2poly* in the file *mbpol.cpp* only. Note the run time is assessed for **each sample**, as the code can only take in one sample to make the calculation. So, the time comparable with what is recored in the #1 is *run_time_per_sample[=Time_per_run] x sample_count[=42105 samples]*

<br>

Code Language  |  Machine  |  GPU ID  |  Iterations |  Total time [s]  |  **Time per run [s]**  |  GPU Type  |  Driver Version  |  Compiler  |  Compiler version  |  Supporting Lib 1  |  Supporting Lib 1 version  |  Supporting Lib 2  |  Supporting Lib 2 version  |  Supporting Lib 3  |  Supporting Lib 3 version  |  Supporting Lib 4  |  Supporting Lib 4 version
 --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
C++ CUDNN  |  Huey  |  0  |  1000  |  21.15  |  0.0212  |  GTX 680  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.61  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Huey  |  1  |  1000  |  21.15  |  0.0212  |  GTX 680  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.61  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Donald  |  0  |  1000  |  19.60  |  0.0196  |  Tesla K40c  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.61  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Donald  |  1  |  1000  |  31.45  |  0.0315  |  GTX 680  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.61  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Chinotto  |  0  |  1000  |  20.84  |  0.0208  |  GTX 1080  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.44  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Chinotto  |  1  |  1000  |  19.86  |  0.0199  |  GTX 1080  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.44  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Chinotto  |  2  |  1000  |  15.93  |  0.0159  |  Tesla K40c  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.44  |  CUDNN  |  6.0.21  |    |    |    |  
C++ CUDNN  |  Chinotto  |  3  |  1000  |  18.79  |  0.0188  |  GTX 680  |  375.26  |  g++  |  4.8.5  |  CUDA  |  8.0.44  |  CUDNN  |  6.0.21  |    |    |    |  
Jupyter Notebook Keras/THEANO fastrun, CUDA enabled, CUDNN enabled  |  Heuy  |  0  |  1000  |  271.95  |  0.2720  |  GTX 680  |  375.26  |  Python   |  3.6.1  |  CUDA  |  8.0.61  |  CUDNN  |  6.0.21  |  Keras  |  2.0.5  |  Theano  |  0.9.0-dev
Jupyter Notebook Keras/THEANO fastrun, CUDA enabled, CUDNN disabled  |  Heuy  |  0  |  1000  |  292.53  |  0.2925  |  GTX 680  |  375.26  |  Python   |  3.6.1  |  CUDA  |  8.0.61  |    |    |  Keras  |  2.0.5  |  Theano  |  0.9.0-dev
Jupyter Notebook Keras/THEANO fastrun, CPU  |  Heuy  |    |  1000  |  521.82  |  0.5218  |    |    |  Python   |  3.6.1  |    |    |    |    |  Keras  |  2.0.5  |  Theano  |  0.9.0-dev
Polynomial 4d algorithem in C++, intel compiler, xHost optimized, openmp enabled  |  skylate  |    |  1000  |  0.05  |  0.000050127 per sample, 2.110597335 for 42105 samples  |    |    |  icpc -xHost  |  17.0.1  |  fopenmp  |    |    |    |    |    |    |
