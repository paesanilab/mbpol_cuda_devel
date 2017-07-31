#if !defined(_READHDF5_H_)
#define _READHDF5_H_



#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <H5Cpp.h>
#include "whichtype.hpp"

#define SPECCHAR "/"


using namespace std;
using namespace H5;




// Function to read in data from HDF5 file by dataset name
template <typename T>
void Read_Layer_Data_By_DatName (H5File file, const char* name, T* & data_out, hsize_t& rank, hsize_t* & dims){
/*
     file      : the opened H5File
     name      : dataset name (include path)
     data_out  : pointer to the first read out data    (returned)
     rank      : dimensionality of data               (returned)
     dims      : size of each dimension of data        (returned)
     
     Returned data_out array is saved in contiguously in memory with size dims[0]*dims[1]*...*dims[rank-1]
     If loaded data is a 2D array, then it is saved in common row-major format
*/
   try{      
         // access the required dataset by path and data set name
         DataSet dset = file.openDataSet(name);

         // get the dataspace
         DataSpace dspace = dset.getSpace();

         // get the dataset type class
         H5T_class_t type_class = dset.getTypeClass();
         
         // following will print out the type name:
         try{
              switch(type_class){
                    case H5T_INTEGER :{
                         cout << name << " : " <<endl;
                         cout <<  "     - type is : H5T_INT " <<endl;
                         IntType intype = dset.getIntType();
                         
                         // Get order of datatype and print message if it's a little endian.
                         H5std_string order_string;
                         H5T_order_t order = intype.getOrder( order_string );
                         cout <<  "     - encoded as : " << order_string << endl;
                         
                         // Get size of the data element stored in file and print it.
                         size_t size = intype.getSize();
                         cout <<  "     - size of each data is : " << size << endl;
                         
                         }break;
                         
                    case H5T_FLOAT :{
                         cout << name << " : " <<endl;
                         cout <<  "     - type is : H5T_FLOAT " <<endl;
                         FloatType fttype = dset.getFloatType();

                         
                         // Get order of datatype and print message if it's a little endian.
                         H5std_string order_string;
                         H5T_order_t order = fttype.getOrder( order_string );
                         cout <<  "     - encoded as : " << order_string << endl;     
                         
                         // Get size of the data element stored in file and print it.
                         size_t size = fttype.getSize();
                         cout <<  "     - size of each data is : " << size << endl;                       
                                           
                         }break;
  
                    default   :{
                         cout << " Don't know data type!!! " <<endl;
                         }break;
              }

         } catch (...){
               cout << " Don't know data type!!! " <<endl;
               // not implemented.
         }

         
         // get the size of the dataset
         rank = dspace.getSimpleExtentNdims(); 
         if(dims){
               delete[] dims;
         }
         dims = new hsize_t[rank] ;
         dspace.getSimpleExtentDims(dims, NULL);
          
          
         for (int ii=0; ii<rank; ii++) {
               cout << "     - dim " << ii << " is of size "<< dims[ii] << endl;  
         }
         

         // Define the memory dataspace
          hsize_t rankm = 1;
          hsize_t dimsm = 1;
          for (int ii=0; ii<rank; ii++){
               dimsm *= dims[ii];
          }


          DataSpace memspace ( rankm, &dimsm );
          if(data_out){
               delete[] data_out;
          }
          data_out = new T[dimsm];
          
          if (TypeIsFloat<T>::value) {
               dset.read( data_out, PredType::NATIVE_FLOAT, memspace, dspace);
          } else if (TypeIsDouble<T>::value){
               dset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dspace);
          } else {
               cout << "ERROR : Support only single and double float at present!" << endl;
               cout << " Read data as single precision float " <<endl;
               dset.read( data_out, PredType::NATIVE_FLOAT, memspace, dspace);
          }
          
     }
     catch(...){
          // not implement
     }
     
     return;
}




// Function to read out group attribute (saving the list of all layer_names and weight names) from a HDF5 file 
vector<string> Read_Attr_Data_By_Seq (H5File file, const char* path, const char* _attr){
/*
     file      : the opened H5File
     path      : path () to the group
    _attr      : the attribute  
  layer_names  : string vector containing all attribute data in creation order  (returned)
    
*/
     vector<string> attr_names;
          
     // Testing obtain the attribute list
     try{
          // Find the group where the model weights saved
          Group* grp = new Group(file.openGroup(path));
          
          // Find attribute of "layer_names"
          Attribute* attr = new Attribute(grp->openAttribute(_attr));

          // get the attribute datatype
          DataType *type = new DataType(attr->getDataType());

          // test the attribute type class
          H5T_class_t type_class = attr->getTypeClass();
          if (!type_class == H5T_STRING){
               cout << " Saved attribute is not string !!! Only string_read is implemented !!!" <<endl;
               file.close();
               throw runtime_error("error reading attr - attribute is not string!");
          }          
          // test if it is variable string
          if(type->isVariableStr()) {
               cout << "Attribute is variable string. Not implemented!" << endl;;
               file.close();
               throw runtime_error("error reading attr - attribute string is variable!");
          }


          // get the attribute dataspace
          DataSpace *dsp = new DataSpace(attr->getSpace());
     
          // get attribute data's rank, dims, and size
          hsize_t  attr_rank = dsp->getSimpleExtentNdims();  // expect rank = 1
          hsize_t *attr_dims = new hsize_t[attr_rank];
          dsp->getSimpleExtentDims(attr_dims, NULL); 
          hsize_t  attr_size = type->getSize();
          
          // count of data
          hsize_t  attr_count = 1;
          for (int ii=0; ii< attr_rank; ii++){
               attr_count *= attr_dims[ii];
          }

          // the attrbute_data buff in contigious memeory
          // note the length of each name is size+1, as in C++ each string has one byte ending
          char*  attr_buff    = new char [attr_count * (attr_size+1)]; 
          
          try{
               StrType str_type(PredType::C_S1, (attr_size+1)); // output string type
               attr->read(str_type, (void*)attr_buff); // readout  
               
               //Buff to output vector
               char* test = attr_buff;
               for (int ii=0; ii< attr_count; ii++, test+=(attr_size+1)){
                  //cout << test <<endl;
                  attr_names.push_back(test);
               } 
          }
          catch(...){
               file.close();
               delete[] attr_buff;
               delete[] attr_dims;
               delete dsp;
               delete type;
               delete attr;
               delete grp;               
          }           
          

          // delete allocated memory
          delete[] attr_buff;
          delete[] attr_dims;
          delete dsp;
          delete type;
          delete attr;
          delete grp;
          
     } catch(...) {
          file.close();
     }

     return attr_names;
}


// concat two string for a path
string mkpath(string _a, string _b){
     string finalstring;
     
     // check if _a ends with "/" , or _b starts with "/" .
     int _a_end_with_char = _a.compare((_a.length()-1),1,SPECCHAR);
     int _b_start_with_char = _b.compare(0,1,SPECCHAR);
     
     if (_a_end_with_char == 0 ){
          if ( _b_start_with_char ==0) {
               // if _a ends with "/" and _b starts with "/"
               _a.pop_back(); // delete last char of _a
          }     
     } else {
          if ( _b_start_with_char !=0) {
               // if _a does not end with "/", and _b does not start with "/"
               _a.push_back(*SPECCHAR); // add a char to _a
          } 
     }
     
     finalstring= _a + _b;

     return finalstring;
}

#endif
