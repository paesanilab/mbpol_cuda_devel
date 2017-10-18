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

template <typename T>
class Gfunction_t{
private:
// Matrix Elementary functions
T cutoff(T R, T R_cut = 10);
T get_cos(T Rij, T Rik, T Rjk);
T get_Gradial(T Rs, T eta, T  Rij);
T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd);

// Vectorized functions
void cutoff(T* & Rdst, T* & Rrsc, size_t n, T R_cut=10);
void get_cos(T * & Rdst, T * & Rij, T * & Rik, T * & Rjk, size_t n);
void get_Gradial(T* & Rdst, T* & Rij, size_t n, T Rs, T eta);
void get_Gangular(T* & Rdst, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );

void get_Gradial_add(T* & Rdst, T* & tmp, T* & Rij, size_t n, T Rs, T eta);
void get_Gangular_add(T* & Rdst, T*& tmp, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );


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






// Methods:
Gfunction_t();
~Gfunction_t();


// load distance matrix
void load_distfile(const char* distfile, int titleline=0);
void load_distfile(const char* distfile, int titleline=0, int threshold_col=0, T threshold=std::numeric_limits<T>::max());

// load distance matrix index 
void load_dist_colidx(const char* colidxfile);           // load model setup ( not implemented yet; need further reference on how to define it ).
void load_dist_colidx_default();   // load default model setup

// load parameter file
void load_paramfile(const char* paramfile);
void load_paramfile_default();     // load default parameter files: H_rad, H_ang, O_rad , O_ang

// load sequence file
void load_seq(const char* seqfile);

// G-function Construction
void make_G();
void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile);

// G-function Normalization
void norm_rows_in_mtx_by_col_vector(T* & src_mtx, size_t src_row, size_t src_col, T* & scale_vec, int offset=0);
size_t get_count_by_percent(T* src, size_t src_count, T percentage);
};

#endif
