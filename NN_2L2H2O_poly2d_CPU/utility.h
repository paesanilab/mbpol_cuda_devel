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

#include "atomTypeID.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else
//#include <gsl/gsl_cblas.h>
#endif


#ifdef _OPENMP
#include <omp.h>
#endif 


//==============================================================================
// A 2D-array type based on vector
template<typename T>
using matrix_by_vector_t = std::vector<std::vector<T> >;




//==============================================================================
//
// Check if a string is a float number
template <typename T>
bool IsFloat( std::string myString ) {
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
bool init_mtx_in_mem(T** & data, size_t rows, size_t cols){
     try{          
          //clearMemo<T>(data);
          if( rows*cols >0) {          
               T * p = new T[rows*cols]();
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
          if (rows > 0){
               cols=mtx[0].size();
               
               init_mtx_in_mem<T>(data, rows, cols);  
               
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(data, mtx, rows)
               #endif                                      
               for(int ii=0; ii<rows; ii++){
                    copy(mtx[ii].begin(), mtx[ii].end(), data[ii]);       
               }          
          } else {
               std::cout << " No Data is read from file as 2D array" << std::endl;
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
                    int checkcol = onelinedata.size() ;                                                        
                    if ( thredhold_col >=0) {           
                         // when thredhold_index is non-negative, check the colum VS max
                         checkcol = thredhold_col;
                    } else {
                         // when thredhold_index is negative, check the column from the end VS max
                         checkcol += thredhold_col;                                        
                    }                         
                    if (onelinedata[thredhold_col] > thredhold_max) {
                         continue;  // If the data exceeds thredhold, ignore this line.
                    }
                    mtx.push_back(onelinedata);                               
               }                           
          }          

          rows=mtx.size();
          if (rows > 0){
               cols=mtx[0].size();
               
               init_mtx_in_mem<T>(data, rows, cols);  
               
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(data, mtx, rows)
               #endif                                      
               for(int ii=0; ii<rows; ii++){
                    copy(mtx[ii].begin(), mtx[ii].end(), data[ii]);       
               }          
          } else {
               std::cout << " No Data is read from file as 2D array" << std::endl;
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
void transpose_mtx(T** & datdst,  T** datrsc,  size_t nrow_rsc, size_t ncol_rsc)
{
     try{ 
     
          if ( datdst== nullptr) init_mtx_in_mem<T>(datdst, ncol_rsc, nrow_rsc);
          
          // best way to transpose a c++ matrix is copying by row    
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(datrsc, datdst, nrow_rsc, ncol_rsc)
          #endif      
          for(int irow = 0; irow< nrow_rsc; irow++){          
               for(int icol=0; icol < ncol_rsc; icol++){
                    datdst[icol][irow] = datrsc[irow][icol];      // not a smart way.
               }
          }          
          
     } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
     }
};




#if defined (_USE_GSL) || defined (_USE_MKL)
// Using cblas_dcopy and cblas_scopy if cblas libraries are employed
template <>
void transpose_mtx<double>(double** & datdst, double** datrsc,  size_t nrow_rsc, size_t ncol_rsc);

template <>
void transpose_mtx<float>(float** & datdst,  float** datrsc, size_t nrow_rsc, size_t ncol_rsc);

#endif



//===============================================================================                                     
//
// Matrix normalization utility functions
size_t get_count_by_percent(size_t src_count, double percentage);
                                     
                                     
template<typename T>
void get_max_each_row(T*& rst,  T* src,  size_t src_rows, size_t src_cols, long int col_start=0, long int col_end=-1){ 
          if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive        
          if(rst == nullptr) rst = new T[src_rows]();               
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
          #endif   
          for(size_t ii=0 ; ii< src_rows ; ii++ ){
               for(size_t jj=col_start; jj<= col_end ; jj++){
                    if ( rst[ii] < abs(src[ii*src_cols + jj]) ) {
                         rst[ii] = abs(src[ii*src_cols + jj]);
                    };
               }
          };     
};


#if defined (_USE_GSL) || defined (_USE_MKL)

template<>
void get_max_each_row<double>(double*& rst, double* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end);

template<>
void get_max_each_row<float>(float*& rst, float* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end);
#endif


template<typename T>
void norm_rows_in_mtx_by_col_vector(T* src_mtx, size_t src_rows, size_t src_cols, T* scale_vec, long int col_start=0, long int col_end=-1 ){
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     // scale each row (from offset index) in a matrix by a column vector
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(src_mtx, src_rows, src_cols, scale_vec, col_start, col_end)
     #endif
     for(int i = 0; i< src_rows; i++){     
          T scale = 1 / scale_vec[i];
          for(int j=col_start; j<= col_end; j++){
               src_mtx[src_cols*i + j] = src_mtx[src_cols*i + j] * scale ;
          }
     }
}


#if defined (_USE_GSL) || defined (_USE_MKL)
template<>
void norm_rows_in_mtx_by_col_vector<double>(double* src_mtx, size_t src_rows, size_t src_cols, double* scale_vec, long int col_start, long int col_end);

template<>
void norm_rows_in_mtx_by_col_vector<float>(float* src_mtx, size_t src_rows, size_t src_cols, float* scale_vec, long int col_start, long int col_end);
#endif


template<typename T>
void norm_rows_by_maxabs_in_each_row(T* src_mtx, size_t src_rows, size_t src_cols, long int max_start_col=0, long int max_end_col=-1, long int norm_start_col =0, long int norm_end_col=-1){     
     
     T* norms = new T[src_rows]();     
     get_max_each_row<T>(norms, src_mtx, src_rows, src_cols, max_start_col, max_end_col);   
     norm_rows_in_mtx_by_col_vector<T>(src_mtx, src_rows, src_cols, norms, norm_start_col, norm_end_col);          
     delete[] norms;
}






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
