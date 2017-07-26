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
#include "testcase_forCUDA_single.in"

#define PATHTOMODEL "/model_weights"    // usual path to the group saving all the layers in HDF5 file
#define LAYERNAMES  "layer_names"       // Attribute name saving the list of layer names in HDF5
#define WEIGHTNAMES "weight_names"      // Attribute name saving the list of weight names in HDF5

#define ACTLINEAR    ActType_t::LINEAR             // Type of activiation layer
#define ACTTANH      ActType_t::TANH

#define DATATYPE    float


using namespace std;
using namespace H5;

int main(int argc, char **argv){

     string ifn = "32_2b_nn_single.hdf5";


     // initialize memory for rank, dims, and data
     // !!! Don't forget to free memory before exit!!!
     hsize_t data_rank=0;
     hsize_t* data_dims = nullptr;
     DATATYPE* data = nullptr;     
     
     hsize_t bias_rank=0;
     hsize_t* bias_dims = nullptr;
     DATATYPE* bias = nullptr;
     
     Layer_Net_t<DATATYPE> layers;
     
     // Open HDF5 file handle, read only
     H5File file(ifn.c_str(),H5F_ACC_RDONLY);
     
     
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
                    

                    
                    if (wt.compare((wt.length()-1),1,"W" )==0){
                         // get out weight data
                         Read_Layer_Data_By_DatName<DATATYPE> (file, datasetPath.c_str(), data, data_rank, data_dims); 
                         
/*                         
                         // tester printing out 
                         int count =1;                         
                         cout << "float " << layername << "_W[" << data_dims[0] << "][" << data_dims[1] << "] = \\" <<endl;   
                         for (int ii=0; ii<data_rank; ii++){
                              count *= data_dims[ii];
                         }                         
                         cout << "{ { " ;
                         for (int ii=0; ii<count ; ii++){
                              cout << setw(8) << data[ii] << " ," ;
                              if( (ii+1) % data_dims[1] ==0 ){
                                   if ( (ii+1) < count ) {
                                        cout << "} ," << endl;
                                        cout << "  { " ;
                                   } else {
                                        cout << "} };" << endl;
                                   }
                              }
                         }
                         cout << endl;                         
*/                          
                    }else{
                         // get out weight data
                         Read_Layer_Data_By_DatName<DATATYPE> (file, datasetPath.c_str(), bias, bias_rank, bias_dims);    
                                             
/*                         
                         // tester printing out   
                         int count =1;
                         cout << "float " << layername << "_b[" << bias_dims[0] << "] = \\" <<endl;
                         for (int ii=0; ii<bias_rank; ii++){
                              count *= bias_dims[ii];
                         }                         
                         cout << "{  " ;
                         for (int ii=0; ii<count ; ii++){
                              cout << setw(8) << bias[ii] << " ," ;
                              if( (ii+1) % bias_dims[0] ==0 ){
                                   if ( (ii+1) < count ) {
                                        cout << "} ," << endl;
                                        cout << "  { " ;
                                   } else {
                                        cout << "}; " << endl;
                                   }
                              }
                         }
                         cout << endl;                             
*/                    
                    }
                    

               }
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
          layers.get_layer_by_seq(12) -> acttype = ACTLINEAR;
          
          cout << endl;
          cout << "Prediction all samples : " <<endl;
          layers.predict(X[0], 11, 69);
          
     } catch (...){
          if(bias!=NULL)       delete[] bias;
          if(bias_dims!=NULL)  delete[] bias_dims;
          if(data!=NULL)       delete[] data;
          if(data_dims!=NULL)  delete[] data_dims;  
          file.close();
     }

     // Free memory of allocated arraies.
     if(bias!=NULL)       delete[] bias;
     if(bias_dims!=NULL)  delete[] bias_dims;
     if(data!=NULL)       delete[] data;
     if(data_dims!=NULL)  delete[] data_dims;     
     file.close();
     return 0;
}
