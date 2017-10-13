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

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
//#include <gsl/gsl_cblas.h>
#endif


#define FLAGSTART '-'
#define FLAGASSGN '='


using namespace std;

//==============================================================================
//
// Memory clear up
void clearMemo(double** & data){
     if (data != nullptr) {
          if (data[0] != nullptr)  delete[] data[0];   
          delete[] data;
     };
     data = nullptr;
     return;
}



void clearMemo(std::vector<double**> & data){
     for (auto it=data.rbegin(); it!=data.rend(); it++){
          clearMemo(*it);
          data.pop_back();
     }
     return;
}


void clearMemo(std::map<std::string, double**> & data){
     for (auto it=data.begin(); it!=data.end(); it++){
          clearMemo(it->second);
          it->second = nullptr;
     }
     return;
}


//==============================================================================
//
// Initialize a matrix in consecutive memory
bool init_mtx_in_mem(double** & data, size_t& rows, size_t& cols){
     try{
          if( rows*cols >0) {          
               double * p = new double[rows*cols];
               data = new double* [rows];
               #ifdef _OPENMP
               #pragma omp parallel for shared(data, p, rows, cols)
               #endif 
               for(int ii=0; ii<rows; ii++){                    
                    data[ii]=p+ii*cols;     
                    memset(data[ii], 0, sizeof(double)*cols);       
               }          
          }     
          return true;
     } catch (...){
          clearMemo(data);   
          return false;
     }
}


//==============================================================================
//
// Check if a string is a float number
bool IsFloat( string& myString ) {
    std::istringstream iss(myString);
    double f;
    iss >> skipws >> f; // skipws ignores leading whitespace
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail(); 
}


//==============================================================================
//
// Read in a 2D array from file and save to  **data / rows / cols
int read2DArrayfile(double** & data, size_t& rows, size_t& cols, const char* file, int titleline, int thredhold_col, double thredhold_max){
    try { 
          
          clearMemo(data);
          
          ifstream ifs(file);
          string line;
          matrix_by_vector_t mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          vector<double> onelinedata;
          while(getline(ifs, line)){          
               char* p = &line[0u];
               char* end;              
               onelinedata.clear();                              
               for( double d = strtod(p, &end); p != end; d = strtod(p, &end) ) {
                    p = end;                    
                    onelinedata.push_back(d);           
               };               
               if(onelinedata.size()>0) mtx.push_back(onelinedata);              
          }          

          rows=mtx.size();
          cols=mtx[0].size();
          
          init_mtx_in_mem(data, rows, cols);  
          
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


//==============================================================================
//
// Matrix transpose
void transpose_mtx(double** & datrsc, double** & datdst, size_t& nrow_rsc, size_t& ncol_rsc){
     try{ 
     
          if ( datdst== nullptr) init_mtx_in_mem(datdst, ncol_rsc, nrow_rsc);
     
         
          // Switch row-col to col-row
          /*
          for(int ii=0; ii<nrow_rsc; ii++){
               for(int jj=0; jj<ncol_rsc; jj++){
                    datdst[jj][ii] = datrsc[ii][jj];
               }
          }*/
          
          // best way to transpose a c++ matrix is copying by row    
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(datrsc, datdst, nrow_rsc, ncol_rsc)
          #endif      
          for(int irow = 0; irow< nrow_rsc; irow++){          
               cblas_dcopy( (const int) ncol_rsc, (const double*) &datrsc[irow][0], 1, &datdst[0][irow], (const int) nrow_rsc);          
          }
          
          // Test copying by col
          /*
          for(int icol = 0; icol< ncol_rsc; icol++){          
               cblas_dcopy(nrow_rsc, &datrsc[0][icol], ncol_rsc, &datdst[icol][0], 1);          
          }*/
          
     } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
     }
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

