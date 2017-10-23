
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <string.h>
#include <vector>
#include <map>
#include <memory>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <iterator>

#include "Gfunction.h"
#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"
#include "timestamps.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
//#include <gsl/gsl_cblas.h>
#else 
//#include <gsl/gsl_cblas.h>
#endif


using namespace std;

//===========================================================================================
// Vectorized functions
//
// These functions are not vectorized at the moment, 
// but API are left as vectorized form for consecutive memory utilization and 
// future compatible possibility with other linear algebra libraries.
// 

// Following functions are defined if cblas library is employed.
#if defined (_USE_GSL) || defined (_USE_MKL)

template <>
void Gfunction_t<double>::get_Gradial_add(double* & rst, double* & Rij, size_t n, double Rs, double eta, double* tmp ){  
     get_Gradial(tmp, Rij, n, Rs, eta);
     cblas_daxpy((const int)n, 1.0, (const double*)tmp, 1, rst, 1);     
};

template <>
void Gfunction_t<float>::get_Gradial_add(float* & rst, float* & Rij, size_t n, float Rs, float eta , float* tmp ){      
     get_Gradial(tmp, Rij, n, Rs, eta);
     cblas_saxpy((const int)n, 1.0, (const float*)tmp, 1, rst, 1);     
};


template <>
void Gfunction_t<double>::get_Gangular_add(double* & rst, double* & Rij, double* & Rik, double*&  Rjk, size_t n, double eta, double zeta, double lambd , double* tmp){
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     cblas_daxpy((const int)n, 1.0, (const double *)tmp, 1, rst, 1);
};


template <>
void Gfunction_t<float>::get_Gangular_add(float* & rst, float* & Rij, float* & Rik, float*&  Rjk, size_t n, float eta, float zeta, float lambd , float* tmp ){
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     cblas_saxpy((const int)n, 1.0, (const float *)tmp, 1, rst, 1);
};

#endif




//================================================================================
// tester
/*
int main(int argc, char** argv){ 

     cout << "Usage:  THIS.EXE  DISTANCE_FILE  [-" << FLAG_DISTFILE_HEADLINE << "=1]"
          << "[-" << FLAG_COLUMN_INDEX_FILE  << "=NONE]"  
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
          << endl << endl;


     Gfunction_t gf;     // the G-function
     
     // distance file headline
     int distheadline = getCmdLineArgumentInt(argc, (const char **)argv, FLAG_DISTFILE_HEADLINE);     
     if(distheadline==0) distheadline=1;     // a special line for test case
          
     // column index file
     string colidxfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_COLUMN_INDEX_FILE, colidxfile);
     
     
     // parameter file
     string paramfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_PARAM_FILE, paramfile);
     
     
     // atom order file
     string ordfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_ATOM_ORDER_FILE, ordfile);          
          

     // make_G(distfile, distheadline, column_idx file, param file, order file)
     gf.make_G(argv[1], distheadline, colidxfile.c_str(), paramfile.c_str(), ordfile.c_str());
     // resutls saved in gf.G which is a map<string:atom_type, double**>
     
     
     
     // Show results
     std::cout.precision(std::numeric_limits<double>::digits10+1);  
     for(auto it= gf.G.begin(); it!=gf.G.end(); it++){
          string atom         = gf.model.atoms[it->first]->name;
          string atom_type    = gf.model.atoms[it->first]->type;
          cout << " G-fn : " << atom << " = " << endl;          
          for(int ii=0; ii<3; ii++){
               for(int jj=0; jj<gf.G_param_max_size[atom_type]; jj++){
                    if ((jj>0)&&( jj%3 ==0 ) ) cout << endl;
                    cout << fixed << setw(16) << it->second[jj][ii] << " " ;                       
               }
          cout << endl;
          }               
     }
     
    
     return 0;
}
*/


