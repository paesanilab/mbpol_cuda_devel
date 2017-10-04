# Implementing CPU based Neural Network model 

## General
These codes model the complete workflow of performing prediction based on Neural Network model in C++ and CBLAS libraries.  

  - Currently the code is progressed at the stage that generating the G-function results used as a raw input for NN model.
  - The Code reads out the parameters for different atom type correlations and the distances between each atom, and generates their G-functions, which is preserved as input for NN model prediction.

## File list
- `utility.h/cpp`                  : File for utility functions
- `timestamps.h/cpp`               : Timer class for benchmarking
- `atomTypeID.h/cpp`               : Class manipulates atom name, type, internal id, etc.
- `readGparams.h/cpp`              : Class reads out the atom type correlations parameters.
- `Gfunction.h/cpp`                : Class constructs the G-function

- `tester.cpp`                     : The tester
- `tester.in`                      : The tester input. It constains the distance information for 3 water dimers.
- `H_rad/H_ang/O_rad/O_ang`        : Atom type correlation parameter files
- `Gfn_order.dat`                  : The order of atom correlation parameters when constructing G-function
- `keras_gfn_double.csv`           : Reference tester output result with doulbe floating point precision from Python Keras/Theano 
- `NN_input_2LHO_correctedD6_f64.dat` : Complete input file for all 42K water dimers. Used for benchmarking.
- `run_bench.x`                    : Benchmarking script
   
### For class file `Gfunction.h/cpp` constructing G-function:
This file use `CBLAS` library and may use `openmp` to speed up performance:  
   - It has a member `model` containing atom name, type, and id information for easily accessing atom information
   - It has a member `GP` containing the correlation parameters used for generating the G-function
   - It may employ the `openmp - pragma omp parallel for simd` to speed up the performance
   - Final results are saved in class member `G` 

### For the provided tester:  
This tester shows the way to construct G-function.
   - It may accept `tester.in` as input. This is the first 3 dimers from `NN_input_2LHO_correctedD6_f64.dat`, and is to check the result accuracy with from Python Keras/Theano.
   - It may also accept `NN_input_2LHO_correctedD6_f64.dat`, which is the complete distance file for 42K water dimers. One may use this input for benchmarking.

## To Compile:
To compile the tester EXE:
   - Check the installation path for `cblas` library. Change the included library path and invoked library name. 
   - Run `[OPTIMIZEOPTION=1] make` to create the executable file `gfunction_generate.x`
      - the OPTIMIZE OPTION may be one of the following:
          - ENABLE_XHOST
          - ENABLE_CAVX512  
          - ENABLE_CRAVX512  
          - ENABLE_CRAVX2  
          - ENABLE_CRAVXI  
          - ENABLE_AVX  
          - NO_VECT
          - (None, by default)
   - Run `make clean` to clean old object and executable files.

## To RUN
To run with tester input:
   - `./gfunction_generate.x tester.in --ordfile=Gfn_order.dat`
   - Compare the final output scores with what are from Python Keras/Theano.  
   
To run with benchmarking:
   - In `run_benchmarking.x` file:
      - Modify the compile optimization options `compile_options`. Each option will be run once.
      - Modify tested omp threads `omp_thread_list`. Each OMP_NUM_THREADS will be run once under each optimization option.
      - Modify iterations for each omp threads `itr_each_thread`. Program will be repeated with each OMP_NUM_THREADS
   -  `./run_benchmarking.x` 
  

