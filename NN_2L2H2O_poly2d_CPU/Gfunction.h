#ifndef GFUNCTION_H
#define GFUNCTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams.h"
#include "atomTypeID.h"

#include "timestamps.h"


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


const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;


template <typename T>
class Gfunction_t{
private:

//===========================================================================================
//
// Matrix Elementary functions
//T cutoff(T R, T R_cut = 10);
//T get_cos(T Rij, T Rik, T Rjk);
//T get_Gradial(T Rs, T eta, T  Rij);
//T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd);

T cutoff(T R, T R_cut=10) {
    T f=0.0;
    if (R < R_cut) {    
        //T t =  tanh(1.0 - R/R_cut) ;   // avoid using `tanh`, which costs more than `exp` 
        T t =  1.0 - R/R_cut;        
        t = exp(2*t);
        t = (t-1) / (t+1);                
        f = t * t * t ;        
    }
    return f  ;
}


T get_cos(T Rij, T Rik, T Rjk) {
    //cosine of the angle between two vectors ij and ik    
    T Rijxik = Rij*Rik ;    
    if ( Rijxik != 0 ) {
          return ( ( Rij*Rij + Rik*Rik - Rjk*Rjk )/ (2.0 * Rijxik) );
    } else {
          return  std::numeric_limits<T>::infinity();
    }
}


T get_Gradial(T  Rij, T Rs, T eta){
     T G_rad = cutoff(Rij);     
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij-Rs)*(Rij-Rs) )  )  ;
     }
     return G_rad;
}


T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){    
    T G_ang = cutoff(Rij)*cutoff(Rik)*cutoff(Rjk);    
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
// Vectorized functions list:
//void cutoff(T* & Rdst, T* & Rrsc, size_t n, T R_cut=10);
//void get_cos(T * & Rdst, T * & Rij, T * & Rik, T * & Rjk, size_t n);
//void get_Gradial(T* & Rdst, T* & Rij, size_t n, T Rs, T eta);
//void get_Gangular(T* & Rdst, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );
//void get_Gradial_add(T* & Rdst, T* & tmp, T* & Rij, size_t n, T Rs, T eta);
//void get_Gangular_add(T* & Rdst, T*& tmp, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );


void cutoff(T* rst, T* Rij, size_t n, T R_cut=10) {    
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, R_cut, n)
#endif    
    for (int i=0; i<n; i++){
          rst[i] = cutoff(Rij[i], R_cut);
    }             
};



void get_cos(T * rst, T * Rij, T * Rik, T * Rjk, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk)
#endif
  for (int i=0; i<n; i++){     
     rst[i] = get_cos(Rij[i], Rik[i], Rjk[i]);  
  }
};



void get_Gradial(T* rst, T* Rij, size_t n, T Rs, T eta, T R_cut=10 ){ 
  cutoff(rst, Rij, n, R_cut);
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rs, eta)
#endif  
  for (int i=0; i<n ; i++) {
     //rst[i] = cutoff(Rij[i]);  // Use vectorized cutoff function instead
     if (rst[i] >0){    
          rst[i] *= exp( -eta * ( (Rij[i]-Rs)*(Rij[i]-Rs) )  )  ;
     }
  } 
};



void get_Gradial_add(T* rst, T* Rij, size_t n, T Rs, T eta , T* tmp = nullptr ){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }          
     get_Gradial(tmp, Rij, n, Rs, eta);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (int ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;          
};
 



void get_Gangular(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd ){
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk, eta, zeta, lambd)
#endif
  for (int i=0; i<n; i++){
    rst[i]=get_Gangular(Rij[i], Rik[i], Rjk[i], eta, zeta, lambd);
  }
};


void get_Gangular_add(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd, T* tmp = nullptr ){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }     
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (int ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;
};




// Timers for benchmarking
timers_t timers;
timerid_t id, id1, id2 , id3;


public:

// Variables:
atom_Type_ID_t model;         // Model, save information about atom names, indexes, and types
size_t natom;                 // Number of atoms registered in the model
idx_t** colidx;              // distance matrix column index mapping

T** dist, ** distT;      // distance matrix and transpose
size_t ndimers, ndistcols;    // number of dimers, number of columns in distance matrix

Gparams_t<T> GP;                 // G-fn paramter class

std::map<idx_t, T**> G;   // G-fn matrix
std::map<std::string, std::map<idx_t, int> > G_param_start_idx;   // Index of each parameter in G-fn
std::map<std::string, std::map<idx_t, int> > G_param_size;        // Parameter sizes in G-fn
std::map<std::string, size_t>                G_param_max_size;    // Total size of G-fn






//==============================================================================================
//
// Gfunction class constructor/destructor
//

Gfunction_t(){
     colidx = nullptr;
     dist = nullptr;
     distT = nullptr;
};


~Gfunction_t(){
     clearMemo<idx_t>(colidx);
     clearMemo<T>(dist);
     clearMemo<T>(distT);     
     for(auto it=G.begin() ; it!=G.end(); it++){
          clearMemo<T>(it->second);
     };
};



//================================================================================================
//
// Gfunction utilities 
//
// load distance matrix
//void load_distfile(const char* distfile, int titleline=0);
//void load_distfile(const char* distfile, int titleline=0, int threshold_col=0, T threshold=std::numeric_limits<T>::max());

// load distance matrix index 
//void load_dist_colidx(const char* colidxfile);           // load model setup ( not implemented yet; need further reference on how to define it ).
//void load_dist_colidx_default();   // load default model setup

// load parameter file
//void load_paramfile(const char* paramfile);
//void load_paramfile_default();     // load default parameter files: H_rad, H_ang, O_rad , O_ang

// load sequence file
//void load_seq(const char* seqfile);
//

// load distance matrix
//void load_distfile(const char* _distfile, int _titleline=0){
//     timers.insert_random_timer( id, 0, "Read_distance_file");
//     timers.timer_start(id);
//     int ifread = read2DArrayfile(dist, ndimers, ndistcols, _distfile, _titleline);     
//     transpose_mtx<T>(dist, distT, ndimers, ndistcols);     
//     timers.timer_end(id);
//};



// load distance matrix and filt out samples that exceed a maximum value
void load_distfile(const char* _distfile, int _titleline=0, int _thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max()){
     timers.insert_random_timer( id, 0, "Read_distance_file");
     timers.timer_start(id);
     if (thredhold_max < std::numeric_limits<T>::max() ) {
          int ifread = read2DArray_with_max_thredhold(dist, ndimers, ndistcols, _distfile, _titleline, _thredhold_col, thredhold_max);     
     } else {     
          int ifread = read2DArrayfile(dist, ndimers, ndistcols, _distfile, _titleline);
     }
     transpose_mtx<T>(distT, dist, ndimers, ndistcols);     
     timers.timer_end(id);
     
    //std::cout << " READ in count of dimers = " << ndimers << std::endl;
};



// load distance matrix column index 

void load_dist_colidx(const char* _dist_idx_file){        // not implemented yet
     if( strlen(_dist_idx_file)>0 ) {     
          std::cout << " Loading custom distance matrix colum index is not implemented yet !" << std::endl;         
          load_dist_colidx_default();     
     }  else {
          load_dist_colidx_default();
     }
};    


void load_dist_colidx_default(){ 
     model.load_default_atom_id(colidx, natom);
};





// load parameter matrix 

void load_paramfile(const char* _paramfile){
     if ( strlen(_paramfile) > 0 ) {     
          GP.read_param_from_file(_paramfile, model); 
     } else {
          load_paramfile_default();
     }
};


void load_paramfile_default(){  
     GP.read_param_from_file("H_rad", model); 
     GP.read_param_from_file("H_ang", model);
     GP.read_param_from_file("O_rad", model);
     GP.read_param_from_file("O_ang", model);  
}





// load sequnece file

void load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
          GP.read_seq_from_file(_seqfile, model);
     } else {
          GP.make_seq_default();
     }
};












//=================================================================================
// G-function Construction
//void make_G();
//void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile);

void make_G(){      
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
          
          std::string atom1 = at1->second->name;                   
          idx_t idx_atom1 = at1->first;
          std::string atom1_type = at1->second->type;
          idx_t idx_atom1_type = model.types[atom1_type]->id; 

          //cout << " Dealing with atom : " << atom1 << " ... " << endl;          
          
          if( G_param_max_size.find(atom1_type) != G_param_max_size.end() ){

               T** g;          
               init_mtx_in_mem<T>(g, G_param_max_size[atom1_type] , ndimers);  // initialize in memory g[param, dimer_idx]
               
               T* tmp = new T[ndimers];  // a temporary space for cache
               
               
               timers.insert_random_timer(id2, 2 , "Gfn_rad+ang_all");
               timers.timer_start(id2);            
               
               // for each atom2
               for(auto at2= model.atoms.begin(); at2!=model.atoms.end(); at2++ ){
                    std::string atom2 = at2->second->name;
                    if(atom1 != atom2){
                         
                         idx_t idx_atom2 = at2->first;
                         std::string atom2_type = at2->second->type;
                         idx_t idx_atom2_type = model.types[atom2_type]->id;
                         idx_t idx_atom12 = idx_atom1_type*idx_atom2_type;
                         

                         // Calculate RAD when it is needed
                         if ( G_param_start_idx[atom1_type].find(idx_atom12) != G_param_start_idx[atom1_type].end() ) {                     
                              //cout << atom1 << " - " << atom2 << endl;                    
                              size_t nrow_params =  G_param_size[atom1_type][idx_atom12];
                              unsigned int icol = colidx[idx_atom1][idx_atom2] ; // col index of the distance to retrieve
                         
                              T Rs, eta;                         
                              int idx_g_atom12 = G_param_start_idx[atom1_type][idx_atom12];

                              
                              for(int i=0 ; i< nrow_params; i++){          
                                   Rs   = GP.params[atom1_type][idx_atom12][i][COL_RAD_RS];
                                   eta  = GP.params[atom1_type][idx_atom12][i][COL_RAD_ETA] ;                                                                    
                                   timers.insert_random_timer(id, idx_atom12, "GRadial");
                                   timers.timer_start(id);
                                   get_Gradial_add(g[idx_g_atom12+i], distT[icol], ndimers, Rs, eta, tmp);         
                                   timers.timer_end(id);
                              }   
                         }


                         
                         timers.insert_random_timer(id1, 3, "Gfn_ang_all");
                         timers.timer_start(id1);                      
                         
                         for(auto at3=next(at2,1) ; at3!=model.atoms.end(); at3++){
                              std::string atom3 = at3->second->name;
                              if(atom3 != atom1) {
                                   idx_t idx_atom3 = at3->first;
                                   std::string atom3_type = at3->second->type;
                                   idx_t idx_atom3_type = model.types[atom3_type]->id;
                                   idx_t idx_atom123 = idx_atom12*idx_atom3_type;

                                   if( G_param_start_idx[atom1_type].find(idx_atom123) != G_param_start_idx[atom1_type].end() ) {
                                   
                                        //cout << atom1 << " - " << atom2 << " - " << atom3 << endl;                      
                                        unsigned int icol = colidx[idx_atom1][idx_atom2] ;  // col index of the distance to retrieve
                                        unsigned int icol2 = colidx[idx_atom1][idx_atom3] ; // col index of the distance to retrieve
                                        unsigned int icol3 = colidx[idx_atom2][idx_atom3] ; // col index of the distance to retrieve
                                        size_t nrow_params =  GP.params[atom1_type][idx_atom123].size();                              
                                        
                                        T lambd, zeta, eta;
                                        int idx_g_atom123 = G_param_start_idx[atom1_type][idx_atom123];

                                        for(int i=0 ; i< nrow_params; i++){      
                                             lambd = GP.params[atom1_type][idx_atom123][i][COL_ANG_LAMBD];
                                             eta   = GP.params[atom1_type][idx_atom123][i][COL_ANG_ETA];
                                             zeta  = GP.params[atom1_type][idx_atom123][i][COL_ANG_ZETA];                    
                                             timers.insert_random_timer(id, idx_atom123, "GAngular");
                                             timers.timer_start(id);
                                             get_Gangular_add(g[idx_g_atom123+i], distT[icol], distT[icol2], distT[icol3], ndimers, eta, zeta, lambd, tmp);         
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




void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile, int _thredhold_col=0, T thredhold_max=std::numeric_limits<T>::min()){     

     load_distfile(_distfile, _titleline, _thredhold_col, thredhold_max);     

     load_dist_colidx(_colidxfile);

     load_paramfile(_paramfile);

     load_seq(_ordfile);         

     make_G();
}






//=================================================================================
// Normalization functions
//
// G-function Normalization
void norm_G_by_maxabs_in_first_percent(double percent){
     size_t max_norm_count = (size_t) ndimers*percent;
     for(auto it = G.begin(); it!=G.end(); it++){
          idx_t atom_idx = it->first;
          std::string atom_type = model.atoms[atom_idx]->type;
          size_t   rows = G_param_max_size[atom_type];
          size_t & cols = ndimers;
          norm_rows_by_maxabs_in_each_row<T>(*(it->second), rows, cols, 0, max_norm_count, 0, -1);
     }
}


};



// Functions specialization for a type. Implementations are in .cpp
#if defined (_USE_GSL) || defined (_USE_MKL)

template <>
void Gfunction_t<double>::get_Gradial_add(double* rst, double* Rij, size_t n, double Rs, double eta, double* tmp );

template <>
void Gfunction_t<float>::get_Gradial_add(float* rst, float* Rij, size_t n, float Rs, float eta , float* tmp );

template <>
void Gfunction_t<double>::get_Gangular_add(double* rst, double* Rij, double* Rik, double* Rjk, size_t n, double eta, double zeta, double lambd , double* tmp);

template <>
void Gfunction_t<float>::get_Gangular_add(float* rst, float* Rij, float* Rik, float* Rjk, size_t n, float eta, float zeta, float lambd , float* tmp );

#endif










#endif
