#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <H5Cpp.h>


#include<cuda.h>
#include<cudnn.h>
#include<cublas_v2.h>


#include "readhdf5.hpp"
#include "network.cu"

#include "NN_2L2H2O_poly2d.in"          // input sample data, in 2D array
#define SAMPLECOUNT 11                  // input sample count
#define SAMPLEDIM   69                  // each input sample's dim 

#define PATHTOMODEL "/model_weights"    // usual path to the group saving all the layers in HDF5 file
#define LAYERNAMES  "layer_names"       // Attribute name saving the list of layer names in HDF5
#define WEIGHTNAMES "weight_names"      // Attribute name saving the list of weight names in HDF5

#define ACTLINEAR   ActType_t::LINEAR             // Type of activiation layer
#define ACTTANH     ActType_t::TANH

#define INFILE1     "32_2b_nn_single.hdf5"     // HDF5 files for different precisions
#define INFILE2     "32_2b_nn_double.hdf5"

// char to check if the dataset is weight or bias, as HDF5 may save the dataset name differently due to version diversity
#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"

#define LASTATVID   12                  // The sequence ID of last activiation layer. We need to do something special for it.

#define MAXSHOWRESULT 20                // Max count of result to show

using namespace std;
using namespace H5;


// tester function, including reading HDF5 file, creating layers, and making the prediction.
template <typename T>
void runtester(const char* filename, const char* checkchar, T* input){
     // initialize memory for rank, dims, and data
     // !!! Don't forget to free memory before exit!!!
     hsize_t data_rank=0;
     hsize_t* data_dims = nullptr;
     T* data = nullptr;     
     
     hsize_t bias_rank=0;
     hsize_t* bias_dims = nullptr;
     T* bias = nullptr;
     
     Layer_Net_t<T> layers;
     
     // reserver for results
     unsigned long int outsize = 0; 
     T* output = nullptr;     
     
     // Open HDF5 file handle, read only
     H5File file(filename,H5F_ACC_RDONLY);
     
     
     try{     
          // Get saved layer names
          vector<string> layernames;
          layernames = Read_Attr_Data_By_Seq(file,PATHTOMODEL, LAYERNAMES); 

          for (string layername : layernames) {
               // for one single layer
               // layer's fullpath
               string layerpath = mkpath ( string(PATHTOMODEL),  layername ) ;
               
               // get this layer's dataset names
               vector<string> weights;
               weights = Read_Attr_Data_By_Seq(file,layerpath.c_str(), WEIGHTNAMES);
               
               
               cout << " Reading out layer data: " << layername << endl;
               for (string wt : weights ) {
                    // foe one data set
                    // dataset's path
                    string datasetPath = mkpath(layerpath,wt) ;
                    
                    // check the dataset name's last character to see if this dataset is a Weight or a Bias
                    if (wt.compare((wt.length()-1),1, checkchar )==0){
                         // get out weight data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), data, data_rank, data_dims); 
                    }else{
                         // get out bias data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), bias, bias_rank, bias_dims);             
                    }
               }
               // When reading out a dense layer, a 2d weight matrix is obtained
               // Otherwise, it is a 0d matrix (null)
               if (data_rank==2){
                    cout << " Initialize dense layer : " << layername << endl;
                    layers.insert_layer(layername, data_dims[0], data_dims[1], data, bias);
                    data_rank=0;
                    bias_rank=0;
               } else {
                    cout << " Initialize activiation layer : " << layername << endl;
                    layers.insert_layer(layername, ACTTANH);               
               }

               
               cout << " Layer " << layername << " is initialized. " <<endl <<endl;
          }
          
          cout << "Inserting Layers finished !" <<endl;
          
          // In our test model, we insert 5 (fully_connect + tanh_activiation) layers
          // plus 1 (fully_connect + linear_activiation) layers
          // So change the last activiation layer's type to linear
          layers.get_layer_by_seq(LASTATVID) -> acttype = ACTLINEAR;
          
          cout << endl;
          cout << "Prediction all samples : " <<endl;
          layers.predict(input, SAMPLECOUNT, SAMPLEDIM, output, outsize);
          
          // show up the final score, to check the result consistency
          // first, setup the precision
          if(TypeIsDouble<T>::value) {
               std::cout.precision(std::numeric_limits<double>::digits10+1);
          } else {
               std::cout.precision(std::numeric_limits<float>::digits10+1);;
          }
          std::cout.setf( std::ios::fixed, std::ios::floatfield );        
          
          // then, select how many results will be shown.
          // if too many output, only show some in the beginning and some in the end
          if (outsize <= MAXSHOWRESULT){
               cout << endl << " Final score are :" <<endl;            
                for(int ii=0; ii<outsize; ii++){
                    cout << (output[ii]) << "  " ;
               }         
          } else {
               cout << " Final score ( first " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=0; ii<(MAXSHOWRESULT/2); ii++){
                    cout << (output[ii]) << "  " ;
               }
               cout << endl << " Final score ( last " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=(outsize-MAXSHOWRESULT/2); ii<outsize; ii++){
                    cout << (output[ii]) << "  " ;
               }                         
          }
          cout << endl;        
          
          
     } catch (...){
          if(bias!=NULL)       delete[] bias;
          if(bias_dims!=NULL)  delete[] bias_dims;
          if(data!=NULL)       delete[] data;
          if(data_dims!=NULL)  delete[] data_dims;  
          if(output!=NULL) delete[] output;          
          file.close();
     }

     // Free memory of allocated arraies.
     if(bias!=NULL)       delete[] bias;
     if(bias_dims!=NULL)  delete[] bias_dims;
     if(data!=NULL)       delete[] data;
     if(data_dims!=NULL)  delete[] data_dims;     
     if(output!=NULL) delete[] output;       
     file.close();
     return;
}



int main(void){
     try{
     cout << " Run tester with single floating point precision : " <<endl;
     runtester<float> (INFILE1, CHECKCHAR1, X[0]);
     cout << endl << endl;
     cout << " ================================================= " <<endl << endl;
     cout << " Run tester with double floating point precision : " <<endl;
     runtester<double>(INFILE2, CHECKCHAR2, Y[0]);
     } catch (...) {
          cudaDeviceReset();
          exit(1);     
     }
     cudaDeviceReset();
     exit(0);      
     
     return 0;
}
