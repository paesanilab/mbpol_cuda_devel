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
#include <mkl_cblas.h>
#else 
//#include <gsl/gsl_cblas.h>
#endif


#ifdef _OPENMP
#include <omp.h>
#endif 


#define FLAGSTART '-'
#define FLAGASSGN '='


using namespace std;



//==============================================================================
//
// Matrix transpose
//template <typename T>
//void transpose_mtx(T** & datrsc, T** & datdst, size_t& nrow_rsc, size_t& ncol_rsc)


// Function definition for a specific type should be left in cpp file
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void transpose_mtx<double>(double** & datdst,  double**  datrsc,  size_t nrow_rsc, size_t ncol_rsc){
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
          //for(int icol = 0; icol< ncol_rsc; icol++){          
          //     cblas_dcopy(nrow_rsc, &datrsc[0][icol], ncol_rsc, &datdst[icol][0], 1);          
          //}
          
     } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
     }
};

template <>
void transpose_mtx<float>(float** & datdst, float** datrsc,  size_t nrow_rsc, size_t ncol_rsc){
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
#endif









//===============================================================================                                     
//
// Matrix normalization utility functions
size_t get_count_by_percent(size_t src_count, double percentage){
     return (size_t) src_count*percentage;
}                                     

// Following functions are defined if cblas library is employed.
#if defined (_USE_GSL) || defined (_USE_MKL)

template<>
void get_max_each_row<double>(double*& rst, double* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end){
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     if(rst == nullptr) rst = new double[src_rows]();   
      
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
     #endif   
     for(size_t ii=0 ; ii< src_rows ; ii++ ){
          size_t i = cblas_idamax( (const int)(col_end - col_start + 1), (const double*)(&src[src_cols*ii + col_start]) , 1 ) ;
          rst[ii] = src[ii*src_cols + col_start + i];
     };
};


template<>
void get_max_each_row<float>(float*& rst, float* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end){
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     if(rst == nullptr) rst = new float[src_rows]();          
     
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
     #endif   
     for(size_t ii=0 ; ii< src_rows ; ii++ ){
          size_t i = cblas_isamax( (const int)(col_end - col_start + 1), (const float*)(&src[src_cols*ii + col_start]) , 1 ) ;
          rst[ii] = src[ii*src_cols + col_start + i];
     };
};


template<>
void norm_rows_in_mtx_by_col_vector(double* src_mtx, size_t src_rows, size_t src_cols, double* scale_vec, long int col_start, long int col_end){
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     // scale each row (from col_start[0,1,2...] to col_end[ ...-3, -2,-1]) in a matrix by a column vector
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(src_mtx, src_rows, src_cols, scale_vec, col_start, col_end)
     #endif
     for(int ii = 0; ii< src_rows; ii++){     
          cblas_dscal( (const int)( col_end - col_start + 1 ), (double)(1/scale_vec[ii]), &(src_mtx[ii*src_cols+col_start]), 1);
     }
}

template<>
void norm_rows_in_mtx_by_col_vector(float* src_mtx, size_t src_rows, size_t src_cols, float* scale_vec, long int col_start, long int col_end){
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     // scale each row (from col_start[0,1,2...] to col_end[ ...-3, -2,-1]) in a matrix by a column vector
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(src_mtx, src_rows, src_cols, scale_vec, col_start, col_end)
     #endif
     for(int ii = 0; ii< src_rows; ii++){     
          cblas_sscal( (const int)(col_end - col_start +1), (float)(1/scale_vec[ii]), &(src_mtx[ii*src_cols+col_start]), 1);
     }
}

#endif



//==============================================================================
//
// Command line arguments manipulation
//
// Ignore the symbol(s) a argument starts with
int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {  
        return 0;
    }

    return string_start;
}

// Check if the argument contains a specific flag
bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter(FLAGSTART, argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, FLAGASSGN);            
            
            int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);          
            
            // `strlen` does NOT count the '\0' in the end of the string 
            int length = (int)strlen(string_ref);

            if (length == argv_length && !strncmp(string_argv, string_ref, length))
            {
                bFound = true;
                break;
            }
        }
    }
    return bFound;
}

// Get the argument integer value
int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter( FLAGSTART , argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!strncmp(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == FLAGASSGN) ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }
                bFound = true;
                continue;
            }
        }
    }
    if (bFound)
    {
        return value;
    }
    else
    {
        //printf("Not found int\n");
        return 0; // default return
    }
}


// Get argument string value
bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, string & string_retval)
{
    bool bFound = false;
    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {           
            int string_start = stringRemoveDelimiter(FLAGSTART, argv[i]);
            char *string_argv = (char *)&argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!strncmp(string_argv, string_ref, length))
            {
                string_retval = &string_argv[length+1];
                bFound = true;
                break;
            }
        }
    }
    if (!bFound)
    {
        string_retval = "";
    }
    return bFound;
}



