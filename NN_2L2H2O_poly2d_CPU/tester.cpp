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

#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"
#include "timestamps.h"
#include "Gfunction.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
//#include <gsl/gsl_cblas.h>
#endif

#include <omp.h>

using namespace std;




const char* FLAG_DISTFILE_HEADLINE = "distheadline" ;
const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
const char* FLAG_PARAM_FILE        =    "paramfile" ;
const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;

// tester
int main(int argc, char** argv){ 

     cout << "Usage:  THIS.EXE  DISTANCE_FILE  [-" << FLAG_DISTFILE_HEADLINE << "=1]"
          << "[-" << FLAG_COLUMN_INDEX_FILE  << "=NONE]"  
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
          << endl << endl;

     if (argc < 2) {
          return 0;    
     } 

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
     // resutls saved in gf.G which is a map<string:atom_idx, double**>
     
     
     
     // Show results on test data set
     if ( !strcmp(argv[1] , "test.dat") ) {
          cout << endl;
          cout << "Output tester results for accuracy checking:" << endl;
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
     }
    
     return 0;
}
