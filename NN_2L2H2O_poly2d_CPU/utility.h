#ifndef UTILITY_H
#define UTILITY_H



#include <limits>
#include <cstdlib>
#include <iomanip>
#include <utility>
#include <cstddef>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>
#include <map>
#include <string>

#include "utility.h"
#include "atomTypeID.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
//#include <gsl/gsl_cblas.h>
#else
#include <gsl/gsl_cblas.h>
#endif

//==============================================================================
// A 2D-array type based on vector
template<typename T>
using matrix_by_vector_t = std::vector<std::vector<T> >;




//==============================================================================
//
// Check if a string is a float number
template <typename T>
bool IsFloat( std::string& myString ) {
    std::istringstream iss(myString);
    T f;
    iss >> std::skipws >> f; // skipws ignores leading whitespace
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail(); 
}




//==============================================================================
//
// Memory management functions
//
template <typename T>
void clearMemo(T** & data){
     if (data != nullptr) {
          if (data[0] != nullptr)  delete[] data[0];   
          delete[] data;
     };
     data = nullptr;
     return;
}


template <typename T>
void clearMemo(std::vector<T**> & data){
     for (auto it=data.rbegin(); it!=data.rend(); it++){
          clearMemo<T>(*it);
          data.pop_back();
     }
     return;
}



template <typename T>
void clearMemo(std::map<std::string, T**> & data){
     for (auto it=data.begin(); it!=data.end(); it++){
          clearMemo<T>(it->second);
          it->second = nullptr;
     }
     return;
}







//==============================================================================
//
// Initialize a matrix in consecutive memory
template <typename T>
bool init_mtx_in_mem(T** & data, size_t& rows, size_t& cols){
     try{
          if( rows*cols >0) {          
               T * p = new T[rows*cols];
               data = new T* [rows];
               #ifdef _OPENMP
               #pragma omp parallel for shared(data, p, rows, cols)
               #endif 
               for(int ii=0; ii<rows; ii++){                    
                    data[ii]=p+ii*cols;     
                    memset(data[ii], 0, sizeof(T)*cols);       
               }          
          }     
          return true;
     } catch (...){
          clearMemo<T>(data);   
          return false;
     }
};






//==============================================================================
//
// Read in a 2D array from file and save to  **data / rows / cols
template <typename T>
int read2DArrayfile(T** & data, size_t& rows, size_t& cols, const char* file, int titleline=0){
    try { 
          
          clearMemo<T>(data);
          
          std::ifstream ifs(file);
          std::string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          std::vector<T> onelinedata;
          while(getline(ifs, line)){          
               char* p = &line[0u];
               char* end;              
               onelinedata.clear();                              
               for( T d = strtod(p, &end); p != end; d = strtod(p, &end) ) {
                    p = end;                    
                    onelinedata.push_back(d);           
               };               
               if(onelinedata.size()>0) mtx.push_back(onelinedata);              
          }          

          rows=mtx.size();
          cols=mtx[0].size();
          
          init_mtx_in_mem<T>(data, rows, cols);  
          
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(data, mtx, rows)
          #endif                                      
          for(int ii=0; ii<rows; ii++){
               copy(mtx[ii].begin(), mtx[ii].end(), data[ii]);       
          }          

          mtx.clear();
          return 0;                    
    } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
        return 1;
    }
};


template <typename T>
int read2DArray_with_max_thredhold(T** & data, size_t& rows, size_t& cols, const char* file, int titleline=0, int thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max()){
    try { 
          
          clearMemo<T>(data);
          
          std::ifstream ifs(file);
          std::string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          std::vector<T> onelinedata;
          while(getline(ifs, line)){          
               char* p = &line[0u];
               char* end;              
               onelinedata.clear();                              
               for( T d = strtod(p, &end); p != end; d = strtod(p, &end) ) {
                    p = end;                    
                    onelinedata.push_back(d);           
               };               
               if (onelinedata.size()>0) {   
                                                           
                    if ( thredhold_col >=0) {           
                         // when thredhold_index is non-negative, check the colum VS max
                         if (onelinedata[thredhold_col] > thredhold_max) {
                              continue;
                         }
                    } else {
                         // when thredhold_index is negative, check the column from the end VS max
                         if ( onelinedata[ onelinedata.size() + thredhold_col] > thredhold_max ) {
                              continue;
                         }                                             
                    }
                    mtx.push_back(onelinedata);                               
               }                           
          }          

          rows=mtx.size();
          cols=mtx[0].size();
          
          init_mtx_in_mem<T>(data, rows, cols);  
          
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(data, mtx, rows)
          #endif                                      
          for(int ii=0; ii<rows; ii++){
               copy(mtx[ii].begin(), mtx[ii].end(), data[ii]);       
          }          

          mtx.clear();
          return 0;                    
    } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
        return 1;
    }
};



//========================================================================================
// 2D array transpose
//
template <typename T>
void transpose_mtx(T** & datrsc, T** & datdst, size_t& nrow_rsc, size_t& ncol_rsc)
{
     std::cout<< "Undefined action with this data type" << std::endl;
};

/*
template <>
void transpose_mtx<double>(double** & datrsc, double** & datdst, size_t& nrow_rsc, size_t& ncol_rsc){
     try{ 
     
          if ( datdst== nullptr) init_mtx_in_mem<double>(datdst, ncol_rsc, nrow_rsc);
     
         
          // Switch row-col to col-row         
          //for(int ii=0; ii<nrow_rsc; ii++){
          //     for(int jj=0; jj<ncol_rsc; jj++){
          //          datdst[jj][ii] = datrsc[ii][jj];
          //     }
          //}
          
          // best way to transpose a c++ matrix is copying by row    
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(datrsc, datdst, nrow_rsc, ncol_rsc)
          #endif      
          for(int irow = 0; irow< nrow_rsc; irow++){          
               cblas_dcopy( (const int) ncol_rsc, (const double*) &datrsc[irow][0], 1, &datdst[0][irow], (const int) nrow_rsc);          
          }
          
          // Test copying by col
          //
          //for(int icol = 0; icol< ncol_rsc; icol++){          
          //     cblas_dcopy(nrow_rsc, &datrsc[0][icol], ncol_rsc, &datdst[icol][0], 1);          
          //}
          
     } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
     }
};

template <>
void transpose_mtx<float>(float** & datrsc, float** & datdst, size_t& nrow_rsc, size_t& ncol_rsc){
     try{ 
     
          if ( datdst== nullptr) init_mtx_in_mem<float>(datdst, ncol_rsc, nrow_rsc);

          #ifdef _OPENMP
          #pragma omp parallel for simd shared(datrsc, datdst, nrow_rsc, ncol_rsc)
          #endif      
          for(int irow = 0; irow< nrow_rsc; irow++){          
               cblas_scopy( (const int) ncol_rsc, (const float*) &datrsc[irow][0], 1, &datdst[0][irow], (const int) nrow_rsc);          
          }                  
     } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
     }
};
*/

//==============================================================================
//
// Utility functions dealing with input arguments
//
int stringRemoveDelimiter(char delimiter, const char *string);

bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref);

int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);

bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, std::string & string_retval);

#endif
