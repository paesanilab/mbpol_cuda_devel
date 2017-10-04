
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

#include <cblas.h>
#include <omp.h>

using namespace std;



const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;


//===========================================================================================
//
// Elementary functions 
//

double Gfunction_t::cutoff(double R, double R_cut) {
    double f=0.0;
    if (R < R_cut) {    
        //double t =  tanh(1.0 - R/R_cut) ;   // avoid using `tanh`, which costs more than `exp` 
        double t =  1.0 - R/R_cut;        
        t = exp(2*t);
        t = (t-1) / (t+1);                
        f = t * t * t ;        
    }
    return f  ;
}


double Gfunction_t::get_cos(double Rij, double Rik, double Rjk) {
    //cosine of the angle between two vectors ij and ik    
    double Rijxik = Rij*Rik ;    
    if ( Rijxik != 0 ) {
          return ( ( Rij*Rij + Rik*Rik - Rjk*Rjk )/ (2.0 * Rijxik) );
    } else {
          return  numeric_limits<double>::infinity();
    }
}

double Gfunction_t::get_Gradial(double  Rij, double Rs, double eta){
     double G_rad = cutoff(Rij);     
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij-Rs)*(Rij-Rs) )  )  ;
     }
     return G_rad;
}

double Gfunction_t::get_Gangular(double Rij, double Rik, double Rjk, double eta, double zeta, double lambd){    
    double G_ang = cutoff(Rij)*cutoff(Rik)*cutoff(Rjk);    
    if ( G_ang > 0) {    
          G_ang *=   2 * pow( (1.0 + lambd* get_cos(Rij, Rik, Rjk))/2.0, zeta) 
                     * exp(-eta*  ( (Rij+Rik+Rjk)*(Rij+Rik+Rjk) ) );    
    } 
    return G_ang ;    
}



//===========================================================================================
// Vectorized functions
//
// These functions are not vectorized at the moment, 
// but API are left as vectorized form for consecutive memory utilization and 
// future compatible possibility with other linear algebra libraries.
// 
void Gfunction_t::cutoff(double* & rst, double* & Rij, size_t n, double R_cut) {    
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, R_cut, n)
#endif    
    for (int i=0; i<n; i++){
          rst[i] = cutoff(Rij[i], R_cut);
    }             
// Here are some efforts on vectorizing the function. 
// But since the BLAS does not support elementary functions, 
// the efficiency is not good.
/*
#ifdef _OPENMP
#pragma omp parallel for simd shared(Rdst,n)
#endif
    for (int i=0; i<n; i++){
          Rdst[i] = 1.0;
    }    
    
    double k = -1/R_cut;
    cblas_daxpy( n , k, Rrsc, 1, Rdst, 1);   // tmp = (1.0 - Rrsc[i]/R_cut)
    
    // if R<= R_cut,   dst = tanh(tmp) ^ 3
    // else, dst = 0.0
    
#ifdef _OPENMP
#pragma omp parallel for simd shared(Rdst, Rrsc, R_cut, n)
#endif    
    for (int i=0; i<n; i++){
          if (Rrsc[i] > R_cut){
               Rdst[i]=0.0;          
          } else {
               double t = tanh(Rdst[i]);
               Rdst[i] = t*t*t;
          };
    };
*/
};


void Gfunction_t::get_cos(double * & rst, double * & Rij, double * & Rik, double * & Rjk, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk)
#endif
  for (int i=0; i<n; i++){     
     rst[i] = get_cos(Rij[i], Rik[i], Rjk[i]);  
  }
}

void Gfunction_t::get_Gradial(double* & rst, double* & Rij, size_t n, double Rs, double eta ){ 
  cutoff(rst, Rij, n);
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rs, eta)
#endif  
  for (int i=0; i<n ; i++) {
     //rst[i] = cutoff(Rij[i]);  // Use vectorized cutoff function instead
     if (rst[i] >0){    
          rst[i] *= exp( -eta * ( (Rij[i]-Rs)*(Rij[i]-Rs) )  )  ;
     }
  } 
}

void Gfunction_t::get_Gradial_add(double* & rst, double*& tmp, double* & Rij, size_t n, double Rs, double eta ){      
     get_Gradial(tmp, Rij, n, Rs, eta);
     cblas_daxpy(n, 1.0, tmp, 1, rst, 1);     
}

void Gfunction_t::get_Gangular(double* & rst, double* & Rij, double* & Rik, double*&  Rjk, size_t n, double eta, double zeta, double lambd ){
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk, eta, zeta, lambd)
#endif
  for (int i=0; i<n; i++){
    rst[i]=get_Gangular(Rij[i], Rik[i], Rjk[i], eta, zeta, lambd);
  }
}


void Gfunction_t::get_Gangular_add(double* & rst, double*& tmp, double* & Rij, double* & Rik, double*&  Rjk, size_t n, double eta, double zeta, double lambd ){
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     cblas_daxpy(n, 1.0, tmp, 1, rst, 1);
}


//==============================================================================================
//
// Gfunction class constructor/destructor
//
Gfunction_t::Gfunction_t(){
     colidx = nullptr;
     dist = nullptr;
     distT = nullptr;
};
Gfunction_t::~Gfunction_t(){
     clearMemo(colidx);
     clearMemo(dist);
     clearMemo(distT);     
     for(auto it=G.begin() ; it!=G.end(); it++){
          clearMemo(it->second);
     };
};

//================================================================================================
//
// Gfunction utilities
//

// load distance matrix
void Gfunction_t::load_distfile(const char* _distfile, int _titleline){
     timers.insert_random_timer( id, 0, "Read_distance_file");
     timers.timer_start(id);
     int ifread = read2DArrayfile(dist, ndimers, ndistcols, _distfile, _titleline);     
     transpose_mtx(dist, distT, ndimers, ndistcols);     
     timers.timer_end(id);
};



// load distance matrix column index 
void Gfunction_t::load_dist_colidx(const char* _dist_idx_file){        // not implemented yet
     if( strlen(_dist_idx_file)>0 ) {     
          cout << " Loading custom distance matrix colum index is not implemented yet !" << endl;         
          load_dist_colidx_default();     
     }  else {
          load_dist_colidx_default();
     }
};    

void Gfunction_t::load_dist_colidx_default(){ 
     model.load_default_atom_id(colidx, natom);
};





// load parameter matrix 
void Gfunction_t::load_paramfile(const char* _paramfile){
     if ( strlen(_paramfile) > 0 ) {     
          GP.read_param_from_file(_paramfile, model); 
     } else {
          load_paramfile_default();
     }
};

void Gfunction_t::load_paramfile_default(){  
     GP.read_param_from_file("H_rad", model); 
     GP.read_param_from_file("H_ang", model);
     GP.read_param_from_file("O_rad", model);
     GP.read_param_from_file("O_ang", model);  
}





// load sequnece file
void Gfunction_t::load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
          GP.read_seq_from_file(_seqfile, model);
     } else {
          GP.make_seq_default();
     }
};









// make G-fns
void Gfunction_t::make_G(){      
     timers.insert_random_timer(id3, 1 , "Gf_run_all");
     timers.timer_start(id3);     
               
     for(auto it = GP.seq.begin(); it!=GP.seq.end(); it++) {
          // it->first  = name of atom type ;
          // it->second = vector<idx_t> ;   a vector saving the list of sequence order
          int curr_idx = 0;
          for(auto it2 = it->second.begin(); it2!=it->second.end(); it2++){                
               // *it2 = element relationship index (=atom1_type * atom2_type * ... )
               G_param_start_idx[it->first][*it2]     = curr_idx;      // The start index is the cumulative size of all relationships until now               
               size_t _tmp = GP.params[it->first][*it2].size();                              
               G_param_size[it->first][*it2] = _tmp;
               curr_idx +=  _tmp;        // next relatinoship               
               //cout << it->first << " " << *it2 << " " << curr_idx << endl;
          }          
          G_param_max_size[it->first] = curr_idx;                // max capacity of this atom type
     }
     
     // For each atom1
     //#pragma omp parallel for shared(model, GP, natom, colidx, distT, ndimers, G, G_param_start_idx, G_param_max_size)
     for (auto at1=model.atoms.begin(); at1!=model.atoms.end(); at1++ ) {
          
          string atom1 = at1->second->name;                   
          idx_t idx_atom1 = at1->first;
          string atom1_type = at1->second->type;
          idx_t idx_atom1_type = model.types[atom1_type]->id; 

          //cout << " Dealing with atom : " << atom1 << " ... " << endl;          
          
          if( G_param_max_size.find(atom1_type) != G_param_max_size.end() ){

               double** g;          
               init_mtx_in_mem(g, G_param_max_size[atom1_type] , ndimers);  // initialize in memory g[param, dimer_idx]
               
               double* tmp = new double[ndimers];  // a temporary space for cache
               
               
               timers.insert_random_timer(id2, 2 , "Gfn_rad+ang_all");
               timers.timer_start(id2);            
               
               // for each atom2
               for(auto at2= model.atoms.begin(); at2!=model.atoms.end(); at2++ ){
                    string atom2 = at2->second->name;
                    if(atom1 != atom2){
                         
                         idx_t idx_atom2 = at2->first;
                         string atom2_type = at2->second->type;
                         idx_t idx_atom2_type = model.types[atom2_type]->id;
                         idx_t idx_atom12 = idx_atom1_type*idx_atom2_type;
                         

                         // Calculate RAD when it is needed
                         if ( G_param_start_idx[atom1_type].find(idx_atom12) != G_param_start_idx[atom1_type].end() ) {                     
                              //cout << atom1 << " - " << atom2 << endl;                    
                              size_t nrow_params =  G_param_size[atom1_type][idx_atom12];
                              unsigned int icol = colidx[idx_atom1][idx_atom2] ; // col index of the distance to retrieve
                         
                              double Rs, eta;                         
                              int idx_g_atom12 = G_param_start_idx[atom1_type][idx_atom12];

                              
                              for(int i=0 ; i< nrow_params; i++){          
                                   Rs   = GP.params[atom1_type][idx_atom12][i][COL_RAD_RS];
                                   eta  = GP.params[atom1_type][idx_atom12][i][COL_RAD_ETA] ;                                                                    
                                   timers.insert_random_timer(id, idx_atom12, "GRadial");
                                   timers.timer_start(id);
                                   get_Gradial_add(g[idx_g_atom12+i], tmp, distT[icol], ndimers, Rs, eta);         
                                   timers.timer_end(id);
                              }   
                         }


                         
                         timers.insert_random_timer(id1, 3, "Gfn_ang_all");
                         timers.timer_start(id1);                      
                         
                         for(auto at3=next(at2,1) ; at3!=model.atoms.end(); at3++){
                              string atom3 = at3->second->name;
                              if(atom3 != atom1) {
                                   idx_t idx_atom3 = at3->first;
                                   string atom3_type = at3->second->type;
                                   idx_t idx_atom3_type = model.types[atom3_type]->id;
                                   idx_t idx_atom123 = idx_atom12*idx_atom3_type;

                                   if( G_param_start_idx[atom1_type].find(idx_atom123) != G_param_start_idx[atom1_type].end() ) {
                                   
                                        //cout << atom1 << " - " << atom2 << " - " << atom3 << endl;                      
                                        unsigned int icol = colidx[idx_atom1][idx_atom2] ;  // col index of the distance to retrieve
                                        unsigned int icol2 = colidx[idx_atom1][idx_atom3] ; // col index of the distance to retrieve
                                        unsigned int icol3 = colidx[idx_atom2][idx_atom3] ; // col index of the distance to retrieve
                                        size_t nrow_params =  GP.params[atom1_type][idx_atom123].size();                              
                                        
                                        double lambd, zeta, eta;
                                        int idx_g_atom123 = G_param_start_idx[atom1_type][idx_atom123];

                                        for(int i=0 ; i< nrow_params; i++){      
                                             lambd = GP.params[atom1_type][idx_atom123][i][COL_ANG_LAMBD];
                                             eta   = GP.params[atom1_type][idx_atom123][i][COL_ANG_ETA];
                                             zeta  = GP.params[atom1_type][idx_atom123][i][COL_ANG_ZETA];                    
                                             timers.insert_random_timer(id, idx_atom123, "GAngular");
                                             timers.timer_start(id);
                                             get_Gangular_add(g[idx_g_atom123+i], tmp, distT[icol], distT[icol2], distT[icol3], ndimers, eta, zeta, lambd);         
                                             timers.timer_end(id, true, false);        
                                        } 
                                   } 

                              }                         
                         }
                         
                         timers.timer_end(id1);
                    }     
               }          
               delete[] tmp;
               
               // Save results to G-fn
               G[at1->first] = g;

               timers.timer_end(id2);                                       
          }
     }
     
     
     timers.timer_end(id3);
     timers.get_all_timers_info();
     timers.get_time_collections();
}




void Gfunction_t::make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile){     
     
     load_distfile(_distfile, _titleline);     
     load_dist_colidx(_colidxfile);
     load_paramfile(_paramfile);
     load_seq(_ordfile);         
     make_G();
}


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


