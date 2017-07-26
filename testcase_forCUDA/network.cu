
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
*    - 2017.07.25 by Yaoguang Zhai:     Ver 0.3
*                                       Class now accept single/double precision data
*                                       Create a list of layers tracking the initialization of NN model
*                                       Class now accept multiple sample inputs
*/

#if !defined(_NN_H_)
#define _NN_H_


#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <algorithm>   
#include<cuda.h>
#include<cudnn.h>
#include<cublas_v2.h>

#include"error_util.hpp"
#include"whichtype.hpp"

using namespace std;

// Helper function showing the data on Device
template <typename T>
void printDeviceVector(int size, T* vec_d)
{
    T *vec;
    vec = new T[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(T), cudaMemcpyDeviceToHost);
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << (vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

//===========================================================================================================
// Type of layers
enum class Type_t {
     UNINITIALIZED  = 0 ,
     DENSE          = 1 ,
     ACTIVIATION    = 2 , 
     MAX_TYPE_VALUE = 3
};


// Type of activiation layer
enum class ActType_t {
     NOACTIVIATION  = 0 ,
     LINEAR         = 1 ,
     TANH           = 2 ,
     MAX_ACTTYPE_VALUE = 3
};



// define layers
template <typename T>
struct Layer_t
{
    string name;
    
    Type_t type ;
                    
    ActType_t acttype;               
    
    int inputs;     // number of input dimension
    int outputs;    // number of output dimension
    T *data_h, *data_d;  // weight matrix in host and device
    T *bias_h, *bias_d;  // bias vector in host and device
    
    Layer_t<T>* prev=nullptr;  // list ptr to previous layer
    Layer_t<T>* next=nullptr;  // list ptr to next layer
    
    
    Layer_t<T>() :  name("Default_Layer"), data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), type(Type_t::UNINITIALIZED), acttype(ActType_t::NOACTIVIATION) {};
                
    
    
    // construct dense layer via loaded matrix from host
    Layer_t<T>( string _name, int _inputs, int _outputs, 
          T* _data_h, T* _bias_h)
                  : inputs(_inputs), outputs(_outputs), type(Type_t::DENSE), acttype(ActType_t::NOACTIVIATION)
    {     
        name = _name ;
        data_h = new T[inputs*outputs];       
        bias_h = new T[outputs];
        copy(_data_h, (_data_h+inputs*outputs), data_h);
        copy(_bias_h, (_bias_h+       outputs), bias_h);

        readAllocMemcpy( inputs * outputs, 
                        &data_h, &data_d);
        readAllocMemcpy( outputs, &bias_h, &bias_d);
        
        //cout<< "Layer weights initializing: " << endl;
        //printDeviceVector(inputs * outputs, data_d);
        
        //cout<< "Layer bias initializing: " <<endl;
        //printDeviceVector( outputs, bias_d);
    }
    
    // construct an activation layer
    Layer_t<T>(string _name, int _acttype)
                  : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), type(Type_t::ACTIVIATION)
    {     
          if (_acttype < int(ActType_t::MAX_ACTTYPE_VALUE) ) {
               acttype = static_cast<ActType_t>(_acttype);
          }
          name = _name;
    }    
    
    Layer_t<T>(string _name, ActType_t _acttype)
                  : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), type(Type_t::ACTIVIATION)
    {     
          
          acttype = _acttype;
          
          name = _name;
    }          
    
    
    ~Layer_t<T>()
    { 
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
    
    
private:

     // Allocate device memory from existing data_h
     void readAllocMemcpy(int size, T** data_h, T** data_d)
     {
         int size_b = size*sizeof(T);
         checkCudaErrors( cudaMalloc(data_d, size_b) );
         checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                     size_b,
                                     cudaMemcpyHostToDevice) );
     }    
     
};


// ===========================================================================================================================================
//
// Function to perform forward action on fully connected layer, according to different type of data
// Must be used after CUBLAS handle has been initialized and matrix saved to device.
//
// Return matrix(C) = alpha * matrx(A[_out,_in]) dot matrix(X[_in,N]) + beta * matrix(bias[_out,N])
// _in/out     : layer's in/out dimension    
// N           : number of samples
// alpha/beta  : scalars
//
//
// Tricky Here!!!! Must use [col row] as input towards cublasSgemm()/cublasDgemm instead of [row col], 
// since they use a "column-major" matrix instead of common "row-major" matrix.
// e.g., for a matrix A =  [a11 a12 a13]
//                         [a21 a22 a23]
// in row-major, sequence of cell in memory is [a11 a12 a13 a21 a22 a23]  <- in our case the data are saved in this way
// in column-major, sequence is [a11 a21 a12 a22 a13 a23]
// Therefore, we must claim to CUBLAS that the saved data ([m,n]) has [n] "rows" and [m] "cols" 
// And DO NOT transpose them, as they has already been regarded as a "transposed" matrix in the eye of CUBLAS.


template <typename T>
struct gemm{
     gemm(          cublasHandle_t cublasHandle, int _input_vector_length, int _output_vector_length, int _vector_counts, 
                    void *_weight, void *_inputs, void *_bias,
                    double alpha=1.0, double beta=1.0){
                    cout << " Don't know what to do with this type of data " << endl;
     };
};

template <>
struct gemm<double>{
     gemm<double> (cublasHandle_t cublasHandle, int _input_vector_length, int _output_vector_length, int _vector_counts, 
                    const double *_weight, const double *_inputs, double *_bias,
                    double alpha=1.0, double beta=1.0){
     
                    checkCublasErrors( cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            _output_vector_length, _vector_counts, _input_vector_length, 
                                            &alpha, 
                                            _weight, _output_vector_length,
                                            _inputs, _input_vector_length,
                                            &beta,
                                            _bias, _output_vector_length) );           
     };
};

template <>
struct gemm<float>{
     gemm<float> ( cublasHandle_t cublasHandle, int _input_vector_length, int _output_vector_length, int _vector_counts, 
                    const float *_weight, const float *_inputs, float *_bias,
                    float alpha=1.0, float beta=1.0){

                    checkCublasErrors( cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            _output_vector_length, _vector_counts, _input_vector_length, 
                                            &alpha, 
                                            _weight, _output_vector_length,
                                            _inputs, _input_vector_length,
                                            &beta,
                                            _bias, _output_vector_length) );
     };
};

// ==================================================================================================================================
//
// Network Algorithem class
// cudnn/cublas handles are initialized when this class is constructed.
//
// Notice: one sample data input is a row vector [1 x width].
// and N samples form a matrix [N x w].
// Therefore, the most optimized solution is to using a 2D-Tensor holding the matrix.
// However, cudnn limits the lowest tensor dimension to 3D, so one need an extra dimension (height, or h) to utilize cudnn TensorDesciptor
// In fact, it does NOT matter how to give h and w, but only to make sure  ( h * w == num_of_data_in_one_sample_vector )
//
// Offered methods :
//                       fullyConnectedForward(layer, N, h, w, float/double* srcData, f/d** dstData)
//                       softmaxForward(n, h, w, * srcData, ** dstData)
//                       activiationforward_tanh (n, h, w, * srcData, ** dstData)
//                       activiationforward_ReLU (n, h, w, * srcData, ** dstData)

template <typename T>
class network_t
{
private:

    cudnnDataType_t dataType;      // specify the data type CUDNN refers to, in {16-bit, 32-bit, 64-bit} floating point 

 
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

     // Setting tensor descriptor 4D (=1 in the last dimension)
     void setTensorDesc4D(cudnnTensorDescriptor_t& tensorDesc, 
                         cudnnDataType_t& dataType,
                         int n,    // number of batch samples
                         int h,    // rows of one sample
                         int w)    // cols of one sample
     {
         const int nDims = 4;           
         int dimA[nDims] = {n,h,w,1};
         int strideA[nDims] = {h*w, w, 1,1}; // stride for each dimension
         checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                                 dataType,
                                                 4,
                                                 dimA,
                                                 strideA ) ); 
     }     

      
  public:
    network_t<T>()
    {     
          if ( TypeIsDouble<T>::value ) {  
               dataType = CUDNN_DATA_DOUBLE;
          } else if ( TypeIsFloat<T>::value ) {
               dataType = CUDNN_DATA_FLOAT;          
          } else {
               cout << " Data type is not single/double precision float ! ERROR! " <<endl;
          }
      
        createHandles();    
    };
    ~network_t<T>()
    {
        destroyHandles();
    }
    
    // Resize device memory
    void resize(int size, T **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(T)) );
        checkCudaErrors( cudaMemset(*data, 0, size*sizeof(T)) );        
    }
    
    
    // add bias into the destination Descriptor
    // Note, "cudnnAddTensor" returns error "CUDNN_STATUS_NOT_SUPPORTED" 
    // if the TensorDescs are initialized as 3D Tensor. So they are set to 4D,
    // with one extra dimension (= 1 in size, of course).
    void addBias(const Layer_t<T>& layer, int _n, int _h, int _w, T *dstdata)
    {
        setTensorDesc4D(biasTensorDesc, dataType,  1, _h, _w);
        setTensorDesc4D(dstTensorDesc,  dataType, _n, _h, _w);
        T alpha = 1.0;
        T beta  = 1.0;
        
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    dstdata) );
    }        
    
  
    // Fully connected forwards, using cublas only
    void fullyConnectedForward(const Layer_t<T>& layer,
                          int& n, int& h, int& w,
                          T* srcData, T** dstData)
    {     
        int dim_x = h * w;
        int dim_y = layer.outputs;
        resize(n*dim_y, dstData);
        
            
        // add bias into dstData
        addBias( layer, n, dim_y, 1, *dstData);
        
        // perform forward calculation
        gemm<T>(cublasHandle, dim_x, dim_y, n, layer.data_d, srcData, *dstData);

        h = dim_y; w = 1;      
        
    } 


 
    // Softmax forwards from CUDNN
    void softmaxForward(int n, int h, int w, T* srcData, T** dstData)
    {
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        T alpha = 1.0;
        T beta  = 0.0;
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
    void activationForward_TANH(int n, int h, int w, T* srcData, T** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_TANH,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );         
    
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        T alpha = 1.0;
        T beta  = 0.0;
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
    void activationForward_ReLU(int n, int h, int w, T* srcData, T** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );   
        resize(n*h*w, dstData);

        setTensorDesc(srcTensorDesc, dataType, n, h, w);
        setTensorDesc(dstTensorDesc, dataType, n, h, w);

        T alpha = 1.0;
        T beta  = 0.0;
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }    

};


//===========================================================================================
//
// Model of all layers (as a doubled list), combined with forward prediction
// 
template <typename T>
class Layer_Net_t{
public:
     Layer_t<T>* root = nullptr;
    
     Layer_Net_t<T>(){};
     
     ~Layer_Net_t<T>(){
          Layer_t<T>* curr = nullptr;
          if (root!= NULL) {
               curr = root;
               while(curr->next){curr = curr->next;} ;
               while(curr->prev){
                    curr = curr->prev;
                    delete curr->next;
                    curr->next =nullptr;
               }
               curr = nullptr;
               delete root;
               root = nullptr;
          }
          
     };    
     
     void insert_layer(string &_name, int _inputs, int _outputs, 
          T*& _data_h, T*& _bias_h){
          if (root!=NULL) {
               Layer_t<T>* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _inputs, _outputs, _data_h, _bias_h);
               curr->next->prev = curr;
          } else {
               root = new Layer_t<T>(_name, _inputs, _outputs, _data_h, _bias_h);
          };
     
     };
     void insert_layer(string &_name, int _acttype){
          if (root!=NULL) {
               Layer_t<T>* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype);
               curr->next->prev = curr;
          } else {
               root = new Layer_t<T>(_name, _acttype);
          };
     
     };     
     
     void insert_layer(string &_name, ActType_t _acttype){
          if (root!=NULL) {
               Layer_t<T>* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype);
               curr->next->prev = curr;
          } else {
               root = new Layer_t<T>(_name, _acttype);
          };
     
     };      
     
     
     Layer_t<T>* get_layer_by_seq(int _n){
          Layer_t<T>* curr=root;
          int i = 1;
          
          while( (curr->next != NULL)  && (i<_n) ){
               curr = curr->next;
               i++ ;
          
          
          };
          return curr;
     }
     
     void predict(T* _inputData, int _n, int _w){
        
        if (root != NULL) {
             network_t<T> neural_net;
             int n,h,w;   // number of sampels in one batch ; height ; width 
             
             T *devData_alpha = NULL, *devData_bravo = NULL;  // two storage places (alpha and bravo) saving data flow
            
             n = _n; h = 1; w = _w;   
             
             // initialize storage alpha and save input vector into it
             checkCudaErrors( cudaMalloc(&devData_alpha, n*h*w*sizeof(T)) );
             checkCudaErrors( cudaMemcpy( devData_alpha, _inputData,
                                          n*h*w*sizeof(T),
                                          cudaMemcpyHostToDevice) );
             int seq =1 ;               //   = 1, read alpha and write to bravo
                                        //   =-1, read bravo and write to alpha
             
             cout << " Read in data ... " << endl;              
             //printDeviceVector(n*h*w, devData_alpha);
                                  
             Layer_t<T>* curr = root;
             do{
               cout << " Processing Layer : " << curr->name << endl;
               if ( curr-> type == Type_t::DENSE ) {
                    if ( seq == 1) {              
                         neural_net.fullyConnectedForward((*curr), n, h, w, devData_alpha, &devData_bravo);
                         
                         //cout<< " After Layer : " << curr->name <<endl;
                         //printDeviceVector(n*h*w, devData_bravo);     
                         
                    } else {
                         neural_net.fullyConnectedForward((*curr), n, h, w, devData_bravo, &devData_alpha);
                         
                         //cout<< " After Layer : " << curr->name <<endl;
                         //printDeviceVector(n*h*w, devData_alpha);                          
                    }
                    seq *= -1;
               } else if (curr -> type == Type_t::ACTIVIATION){
                    if (curr -> acttype == ActType_t::TANH){
                    
                         
                         if ( seq == 1) {              
                              neural_net.activationForward_TANH(n, h, w, devData_alpha, &devData_bravo);
                              
                              //cout<< " After Layer : " << curr->name <<endl;
                              //printDeviceVector(n*h*w, devData_bravo);                                                 
                         } else {
                              neural_net.activationForward_TANH(n, h, w, devData_bravo, &devData_alpha);
                              
                              //cout<< " After Layer : " << curr->name <<endl;
                              //printDeviceVector(n*h*w, devData_alpha);                               
                         }
                         seq *= -1;                         
                    } else if (curr->acttype != ActType_t::LINEAR) {
                         cout << "Unknown activation type!" <<endl;
                    } 
               } else {
                    cout << "Unknown layer type!" <<endl;
               }
             } while(  (curr=curr->next) != NULL);
             
             cout << "Final score : " ;         
             if (seq == 1) {
                    printDeviceVector<T>(n*h*w, devData_alpha);
             } else {
                    printDeviceVector<T>(n*h*w, devData_bravo);
             }
             
            
              
             checkCudaErrors( cudaFree(devData_alpha) );
             checkCudaErrors( cudaFree(devData_bravo) );
        
        }
        return;
         
     } 
};





#endif //end of "_NN_H_"
