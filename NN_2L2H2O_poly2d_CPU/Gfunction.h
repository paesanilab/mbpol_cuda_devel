#ifndef GFUNCTION_H
#define GFUNCTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include "readGparams.h"
#include "atomTypeID.h"

#include "timestamps.h"

class Gfunction_t{
private:
// Matrix Elementary functions
double cutoff(double R, double R_cut = 10);
double get_cos(double Rij, double Rik, double Rjk);
double get_Gradial(double Rs, double eta, double  Rij);
double get_Gangular(double Rij, double Rik, double Rjk, double eta, double zeta, double lambd);

// Vectorized functions
void cutoff(double* & Rdst, double* & Rrsc, size_t n, double R_cut=10);
void get_cos(double * & Rdst, double * & Rij, double * & Rik, double * & Rjk, size_t n);
void get_Gradial(double* & Rdst, double* & Rij, size_t n, double Rs, double eta);
void get_Gangular(double* & Rdst, double* & Rij, double* & Rik, double* & Rjk, size_t n, double eta,double zeta, double lambd );

void get_Gradial_add(double* & Rdst, double* & tmp, double* & Rij, size_t n, double Rs, double eta);
void get_Gangular_add(double* & Rdst, double*& tmp, double* & Rij, double* & Rik, double* & Rjk, size_t n, double eta,double zeta, double lambd );


// Timers for benchmarking
timers_t timers;
timerid_t id, id1, id2 , id3;


public:

// Variables:
atom_Type_ID_t model;         // Model, save information about atom names, indexes, and types
size_t natom;                 // Number of atoms registered in the model
double** colidx;              // distance matrix column index mapping

double** dist, ** distT;      // distance matrix and transpose
size_t ndimers, ndistcols;    // number of dimers, number of columns in distance matrix

Gparams_t GP;                 // G-fn paramter class

std::map<idx_t, double**> G;   // G-fn matrix
std::map<std::string, std::map<idx_t, int> > G_param_start_idx;   // Index of each parameter in G-fn
std::map<std::string, std::map<idx_t, int> > G_param_size;        // Parameter sizes in G-fn
std::map<std::string, size_t>                G_param_max_size;    // Total size of G-fn






// Methods:
Gfunction_t();
~Gfunction_t();


// load distance matrix
void load_distfile(const char* distfile, int titleline=0);

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
};

#endif
