
/**
* This file describes Neural Network model class and layer class based on CUDNN/CUBLAS/CUDA library
* 
* It is created from NVIDIA's CUDNN example: "mnistCUDNN.cpp" in which the following desclaimed is contained:
*
***********************************************************************
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
***********************************************************************
*
*
* ===========================================================================================================================
*
* Some concept good to know for CUDNN/CUBLAS:
*    - CUDNN/CUBLAS needs a Handle to initialze its context. This Handle will be passed in all calling to its subfunctions.
*         -- This Handle is used to control library's function on host threads, GPUs and CUDA streams.
*         -- Create this Handle at the beginning, and destroy it after finishing application to release resource.
*    - CUDNN initialize a TensorDescriptor (or in simple words, a matrix with unspecific dimensionality and size) 
*      that helps holds data throughout the Neural Network flow
*         -- This TensorDescriptor needs to be created at beginning, specified dimensionality and size before use, 
*            and destroyed after application
*         -- There are some pre-defined type of TensorDescriptors avaiable in CUDNN, but minimum dimensionality is 3.
*            In our test case, we use the simplest 3d Tensor [batch_num(training samples each time, simply = 1 in toy_tester), x(simply = 1), y]
*    - There are pre-defined Tensor operations used for describing Neural Network layer operation algorithm
*         -- E.g., cudnnActivationForward() will take input Tensor and output its activation resutls.
*         -- However, there is no function describing basic Fully Connected layer algorithm, 
*            as this algorithm is only an utilization of function cublasSgemv() from CUBLAS.
*            This is why CUBLAS is needed.
*
* Hope these explanations will help your understanding of CUDNN.
*
* - By Yaoguang Zhai, 2017.07.14
*
*=============================================================================================================================
*.
* This file is for the purpose of using CUDNN library to implement basic NN models.
* It uses the "error_util.h" file coming from the "mnistCUDNN" example composed by nVIDIA.
*
* Currtent implemented contents include:
*    - Layers, its input/output dimension (m,n), the weighted matrix (W[m,n]), a bias vector (b[n])
*              and a kernal dimension (1X1, for convolution layer)
*    - simple fully connecte forward, using cublasSgemv() function
*    - softmax forwards, using cudnnSoftmaxForward()
*    - Activation_TANH forwards, using cudnnActivationForward() with CUDNN_ACTIVATION_TANH
*    - Activation_ReLU forwards, using cudnnActivationForward() with CUDNN_ACTIVATION_RELU
*
* The code currently works with single precision float.
*
* Future implement:
*    - Double, half precision support
*    - Convolution NN support
*
*=======================================================================
*
*Log:
*    - 2017.07.10 by Yaoguang Zhai:     First version 0.1
*    - 2017.07.14 by Yaoguang Zhai:     Ver 0.2
*                                       Change the TensorDescriptor from 4D to 3D.
*                                       Functions that are not used in test case are removed to reduce learning distraction.
*/

#if !defined(_NN_H_)
#define _NN_H_


#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>

#include<cuda.h>
#include<cudnn.h>
#include<cublas_v2.h>

#include"error_util.hpp"

using namespace std;

// Helper function showing the data on Device
void printDeviceVector(int size, float* vec_d)
{
    float *vec;
    vec = new float[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << (vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

// Helper function getting the loss
__global__ void getLoss(float* dat, float* rst){
     *rst = -logf(*dat);
}

// define layer
struct Layer_t
{
    int inputs;     // number of input dimension
    int outputs;    // number of output dimension
    float *data_h, *data_d;  // weight matrix in host and device
    float *bias_h, *bias_d;  // bias vector in host and device
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0) {};
                
 
    // construct layer via loaded matrix from host
    Layer_t(int _inputs, int _outputs, 
          float* _data_h, float* _bias_h)
                  : inputs(_inputs), outputs(_outputs)
    {
        data_h = _data_h;
        bias_h = _bias_h;
        
        readAllocMemcpy( inputs * outputs, 
                        &data_h, &data_d);
        readAllocMemcpy( outputs, &bias_h, &bias_d);
        
        cout<< "Layer weights initializing: " << endl;
        printDeviceVector(inputs * outputs, data_d);
        
        cout<< "Layer bias initializing: " <<endl;
        printDeviceVector( outputs, bias_d);
        
    }
    
    
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
    
    
private:

     // Allocate device memory from existing data_h
     void readAllocMemcpy(int size, float** data_h, float** data_d)
     {
         int size_b = size*sizeof(float);
         checkCudaErrors( cudaMalloc(data_d, size_b) );
         checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                     size_b,
                                     cudaMemcpyHostToDevice) );
     }    
     
};




class network_t
{
private:

    cudnnDataType_t dataType=CUDNN_DATA_FLOAT;  // specify the data type CUDNN refers to, in {16-bit, 32-bit, 64-bit} floating point 
                                                // In this version, we use only float(32bit)
 
    cudnnHandle_t cudnnHandle;     // a pointer to a structure holding the CUDNN library context
                                   // Must be created via "cudnnCreate()" and destroyed at the end by "cudnnDestroy()"
                                   // Must be passed to ALL subsequent library functions!!!

    cublasHandle_t cublasHandle;   // cublas handle, similar as above cudnn_handle

    
    // Opaque tensor descriptors (N-dim matrix), for operations of layers
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;

    cudnnActivationDescriptor_t  activDesc; // Algorithm used in CUDNN

    
    // create and destroy handles/descriptors, note the sequence of creating/destroying
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) ); 
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCublasErrors( cublasCreate(&cublasHandle) );
    }
    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );
        checkCublasErrors( cublasDestroy(cublasHandle) );
    }

    
     // Function to perform forward action on fully connected layer
     // Return vector(y) = alpha * matrx(A[m,n]) dot vector(X) + beta * vector(y)
     void gemv(cublasHandle_t cublasHandle, int m, int n,  
               const float *A, const float *x, float *y,
               float alpha=1.0, float beta=1.0)
     {    
        // Tricky Here!!!! Must use [col row] as input towards cublasSgemv() instead of [row col], 
        //as cublasSgemv uses a "column-major" matrix instead of common "row-major" matrix.
        // e.g., for a matrix A =  [a11 a12 a13]
        //                         [a21 a22 a23]
        //in row-major, sequence of cell in memory is [a11 a12 a13 a21 a22 a23]  <- in our case the data are saved in this way
        //in column-major, sequence is [a11 a21 a12 a22 a13 a23]
        //Therefore, we must claim to CUBLAS that the saved data has [n] "rows" and [m] "cols"
        // And DO NOT transpose the matrix A!
         checkCublasErrors( cublasSgemv(cublasHandle, CUBLAS_OP_N,  
                                       n, m,
                                       &alpha,
                                       A, n,
                                       x, 1,
                                       &beta,
                                       y, 1) );    
     };    
    

     // Setting tensor descriptor
     void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                         cudnnDataType_t& dataType,
                         int n,    // number of batch samples
                         int h,    // rows of one sample
                         int w)    // cols of one sample
     {
         const int nDims = 3;           
         int dimA[nDims] = {n,h,w};
         int strideA[nDims] = {h*w, w, 1}; // stride for each dimension
         checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                                 dataType,
                                                 3,
                                                 dimA,
                                                 strideA ) ); 
     }


       
  public:
    network_t()
    {
        dataType = CUDNN_DATA_FLOAT;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    
    // Resize device memory
    void resize(int size, float **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(float)) );
    }
    
  
    // Fully connected forwards, using cublas only
    void fullyConnectedForward(const Layer_t& layer,
                          int& n, int& h, int& w,
                          float* srcData, float** dstData)
    {     
        // Test only 1 image! 
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        
        
        int dim_x = h*w;
        int dim_y = layer.outputs;
        resize(dim_y, dstData);
            
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, layer.bias_d, dim_y*sizeof(float), cudaMemcpyDeviceToDevice) );
        
        // perform forward calculation
        gemv(cublasHandle, dim_x, dim_y, layer.data_d, srcData, *dstData);

        h = dim_y; w = 1;      
    } 

 
    // Softmax forwards from CUDNN
    void softmaxForward(int n, int h, int w, float* srcData, float** dstData)
    {
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        float alpha = 1.0;
        float beta  = 0.0;
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_FAST ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    
    // activation forward with hyperbolic tangential
    void activationForward_TANH(int n, int h, int w, float* srcData, float** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_TANH,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        float alpha = 1.0;
        float beta  = 0.0;
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }
    
    // activation forward with ReLU nonlinearty    
    void activationForward_ReLU(int n, int h, int w, float* srcData, float** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        float alpha = 1.0;
        float beta  = 0.0;
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }    

    // toy_test: a simple test case for a two_layer NN model:
    // a fully connected layer + ReLU nonlinety
    // and a fully connected layer + softmax activation
    int toy_test(float* _inputData, int _h, int _w,
                          const Layer_t& layer1,
                          const Layer_t& layer2,
                          const int target_num)
    {
        int n,h,w;   // number of sampels in one batch ; height ; width 
        float *devData_alpha = NULL, *devData_bravo = NULL;  // two storage places (alpha and bravo) saving data flow

        cout<< "Allocate memory ... " << endl;        
       
        // initialize storage alpha and save input vector into it
        checkCudaErrors( cudaMalloc(&devData_alpha, _h*_w*sizeof(float)) );
        checkCudaErrors( cudaMemcpy( devData_alpha, _inputData,
                                     _h*_w*sizeof(float),
                                     cudaMemcpyHostToDevice) );
                                    
        std::cout << "Performing forward propagation in the two layer NN model ...\n";                             
        n = 1; h = _h; w = _w;  
        
        cout<< " Load X input : " << endl;
        printDeviceVector(n*h*w, devData_alpha);  
        
        // Layer 1 : fully connected forward, read from alpha and save result to bravo
        fullyConnectedForward(layer1, n, h, w, devData_alpha, &devData_bravo);
        cout<< " After Layer1 : fullyconnect forward : " <<endl;
        printDeviceVector(n*h*w, devData_bravo); 
     
        // Layer 1 : ReLU forward, read from bravo and save to alpha
        activationForward_ReLU(n, h, w, devData_bravo, &devData_alpha);
        cout<< " After Layer1 : ReLU : " <<endl;
        printDeviceVector(n*h*w, devData_alpha);         
        
        
        // Layer 2 : fully connected forward, read from alpha and save to bravo
        fullyConnectedForward(layer2, n, h, w, devData_alpha, &devData_bravo);
        cout<< " After Layer2 : fullyconnect forward : " <<endl;
        printDeviceVector(n*h*w, devData_bravo);         
        
        // Result after softmax forward, read from bravo and save to alpha
        softmaxForward(n, h, w, devData_bravo, &devData_alpha);
        cout << " Softmax for each classifier : " <<endl;
        printDeviceVector(n*h*w, devData_alpha);
        
     
        // Get the loss of this sample, read from alpha and save to bravo
        resize(1,&devData_bravo);
        getLoss<<<1,1>>>(&(devData_alpha[target_num]),devData_bravo);
        cout<< " loss for this sample is : ";
        printDeviceVector(1, devData_bravo);
        
        
        cout<< " NN forwards finish ! " <<endl;
        
        checkCudaErrors( cudaFree(devData_alpha) );
        checkCudaErrors( cudaFree(devData_bravo) );
        return 0;
    }


};




#endif //end of "_NN_H_"
