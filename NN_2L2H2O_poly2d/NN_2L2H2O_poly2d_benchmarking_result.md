# Benchmarking result of NN_2L2OH2O_poly2d 

### This file contains :  

1. the benchmarking result of `NN_2L2OH2O_poly2d` in Neural Network model on 42105 double-precision samples. This benchmarking is made to the part of the code *predicting the final score* of all samples, excluding *loading parameters, initializing layers and GPU resource context, etc*. The result records the run time on different machine/GPU, supporting library version, driver version, compiler version etc.
2. the benchmarking result of calculating potential energy `E2poly` in polynomial method. It is made on the part of the code *calculating the potential energy E2poly* as the function `poly_2b_v6x::eval` only. Note the run time is assessed for **each sample**, as the code can only take in one sample (double precision) to make the calculation. So, the time comparable with what is recored in the #1 is *run_time_per_sample[=Time_per_run] x sample_count[=42105 samples]*

<br>

Code Language  |  Machine  |  GPU ID  |  GPU Type  |  Iterations  |  Total time [s]  |  Time per run [s]  |  Compiler  
--- | --- | --- | --- | --- | --- | --- | --- |
C++ CUDNN  |  Huey  |  0/1|  GTX 680  - driver 375.26  |  1000  |  21.15  |  0.0212  |  g++[4.8.5] CUDA[8.0.61] CUDNN[6.0.21]
C++ CUDNN  |  Donald  |  0  |  Tesla K40c - driver 375.26  |  1000  |  19.60  |  0.0196  |  g++[4.8.5] CUDA[8.0.61] CUDNN[6.0.21]
C++ CUDNN  |  Donald  |  1  |  GTX 680  - driver 375.26  |  1000  |  31.45  |  0.0315  |  g++[4.8.5] CUDA[8.0.61] CUDNN[6.0.21]
C++ CUDNN  |  Chinotto  |  0/1 | GTX 1080 - driver 375.26  |  1000  |  20.84  |  0.0208  |  g++[4.8.5] CUDA[8.0.44] CUDNN[6.0.21]
C++ CUDNN  |  Chinotto  |  2  |  Tesla K40c - driver 375.26  |  1000  |  15.93  |  0.0159  |  g++[4.8.5] CUDA[8.0.44] CUDNN[6.0.21]
C++ CUDNN  |  Chinotto  |  3  |  GTX 680  - driver 375.26  |  1000  |  18.79  |  0.0188  |  g++[4.8.5] CUDA[8.0.44] CUDNN[6.0.21]
Jupyter Notebook Keras/THEANO fastrun, CUDA enabled, CUDNN enabled  |  Heuy  |  0  |  GTX 680  - driver 375.26  |  1000  |  271.95  |  0.2720  |  Python[3.6.1] CUDA[8.0.61] CUDNN[6.0.21] Keras[2.0.5] Theano[0.9.0-dev]
Jupyter Notebook Keras/THEANO fastrun, CUDA enabled, CUDNN disabled  |  Heuy  |  0  |  GTX 680 - driver 375.26  |  1000  |  292.53  |  0.2925  |  Python[3.6.1] CUDA[8.0.61] Keras[2.0.5] Theano[0.9.0-dev]
Jupyter Notebook Keras/THEANO fastrun, CPU  |  Heuy  |    |    |  1000  |  521.82  |  0.5218  |  Python[3.6.1] Keras[2.0.5] Theano[0.9.0-dev]
Polynomial 4d algorithem in C++, intel compiler, xHost optimized, openmp enabled  |  skylate  |    |    |  1000  |  0.045  |  0.000044772[s] per sample -> estimate: 1.885[s] for 42105 samples  |  icpc[17.0.1] -xHost -fopenmp
