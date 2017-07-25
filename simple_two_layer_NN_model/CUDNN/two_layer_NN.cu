/**
* This is the tester file for a simple two_layer NN model, 
* generating the same forward calculation of a fully_connected layer + ReLU non_linety and a fully connected layer + softmax activation
*
* It uses the same input and weighted matrix as in Python script.
* Inputs and weights are saved in "layer_toy_param.dat" which is loaded directly
*
*/




#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h> 
#include <cudnn.h>
#include "error_util.hpp"       // nVidia's error handler on CUDA/CUDNN/CUBLAS

#include "network.cu"
#include "layer_toy_param.dat"

using namespace std;

int main(int argc, char *argv[])
{   
    int version = (int)cudnnGetVersion();  // display the currunt CUDNN library version
    
    // next three lines are utility functions to check the gpu-devices
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);  // CUDNN_VERSION(_STR) is checked from cudnn.h
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showDevices();
     
    // next block is to set which device is to use
    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
    }
    std::cout << "Using device " << device << std::endl;


   
    // start NN tests of two layers : toy values
    std::cout << "\nTesting loading toy data to check implementation\n";
    
  
    int h=1;  // 5 samples, each is a 1x4 vector data
    int w=4;
    
    network_t test_toy_net;
    
    cout << " network initialized " << endl;
    
    
    Layer_t layer1(4,10, &toy_layer1_W[0][0], toy_layer1_b);
    Layer_t layer2(10,3, &toy_layer2_W[0][0], toy_layer2_b);
    
    cout<<endl << endl<< " Testing sampe NO.1 : " << endl;
    test_toy_net.toy_test(toy_input_1,h,w,layer1,layer2,toy_output_target[0]);
    
    
    cout<<endl << endl<< " Testing sampe NO.2 : " << endl;
    test_toy_net.toy_test(toy_input_2,h,w,layer1,layer2,toy_output_target[1]);
    
    cout<<endl << endl<< " Testing sampe NO.3 : " << endl;
    test_toy_net.toy_test(toy_input_3,h,w,layer1,layer2,toy_output_target[2]);
    
    cout<<endl << endl<< " Testing sampe NO.4 : " << endl;
    test_toy_net.toy_test(toy_input_4,h,w,layer1,layer2,toy_output_target[3]);
    
    cout<<endl << endl<< " Testing sampe NO.5 : " << endl;
    test_toy_net.toy_test(toy_input_5,h,w,layer1,layer2,toy_output_target[4]);
    
    cudaDeviceReset();
    exit(EXIT_SUCCESS);      
}

