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



#define FLAGSTART '-'
#define FLAGASSGN '='


using namespace std;

//==============================================================================
//
// Memory clear up
// These definitions are put into header file
/*
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
*/



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
}

/*
template bool init_mtx_in_mem<double>(double** & data, size_t& rows, size_t& cols);
template bool init_mtx_in_mem<float>(float** & data, size_t& rows, size_t& cols);
template bool init_mtx_in_mem<unsigned int>(unsigned int** & data, size_t& rows, size_t& cols);
template bool init_mtx_in_mem<int>(int** & data, size_t& rows, size_t& cols);
*/


//==============================================================================
//
// Check if a string is a float number
template <typename T>
bool IsFloat( string& myString ) {
    std::istringstream iss(myString);
    T f;
    iss >> skipws >> f; // skipws ignores leading whitespace
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail(); 
}


//==============================================================================
//
// Read in a 2D array from file and save to  **data / rows / cols
template <typename T>
int read2DArrayfile(T** & data, size_t& rows, size_t& cols, const char* file, int titleline){
    try { 
          
          clearMemo<T>(data);
          
          ifstream ifs(file);
          string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          vector<T> onelinedata;
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
}


template <typename T>
int read2DArray_with_max_thredhold(T** & data, size_t& rows, size_t& cols, const char* file, int titleline, int thredhold_col, T thredhold_max){
    try { 
          
          clearMemo<T>(data);
          
          ifstream ifs(file);
          string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          vector<T> onelinedata;
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
}


template <typename T>
void transpose_mtx(T** & datrsc, T** & datdst, size_t& nrow_rsc, size_t& ncol_rsc)
{
     cout<< "Undefined action with this data type" << std::endl;
};

//==============================================================================
//
// Matrix transpose
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



//================================================================================
//
//template functions realization
template <typename T>
void _NEVER_USED_INSTANLIZATION_UTILITY(){
     T** dptr;
     size_t rows =1 , cols=1;     
     init_mtx_in_mem<T>(dptr, rows, cols);

     vector<T**> vec;          
     map<string, T**> mp;     
     
     string mystring = "SOME";
     
     IsFloat<T>(mystring);
     
     read2DArrayfile<T>(dptr, rows, cols, "hello");
     read2DArray_with_max_thredhold<T>(dptr, rows, cols, "Hello", 1, 1, 0);          
     transpose_mtx<T>(dptr, dptr, rows, cols);     
     clearMemo<T>(vec);     
     clearMemo<T>(mp);    
     clearMemo<T>(dptr);
          
}


/*
template class _NEVER_USED_INSTANLIZATION_UTILITY<double> ;
template class _NEVER_USED_INSTANLIZATION_UTILITY<float> ;
template class _NEVER_USED_INSTANLIZATION_UTILITY<unsigned int> ;
template class _NEVER_USED_INSTANLIZATION_UTILITY<int> ;
*/

void NEVER_USED_INSTANLIZATION_UTILITY(){
     _NEVER_USED_INSTANLIZATION_UTILITY<double>();
     _NEVER_USED_INSTANLIZATION_UTILITY<float>() ;
     _NEVER_USED_INSTANLIZATION_UTILITY<idx_t>();
     _NEVER_USED_INSTANLIZATION_UTILITY<int>();
     _NEVER_USED_INSTANLIZATION_UTILITY<size_t>();
}



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



