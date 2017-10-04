#ifndef GPARAMS_H
#define GPARAMS_H

#include <vector>
#include <string>
#include <map>

#include "atomTypeID.h"


// Read parameters for G-function construction
struct Gparams_t {
     std::map<std::string,  std::map<idx_t, std::vector<std::vector<double> > > > params;  
     
     std::map<std::string,  std::vector<idx_t> > seq;
     
     
     int read_param_from_file(const char* p, atom_Type_ID_t & model); // Read in params from a file; save numbers in `params`; save related atom information into `model`.
     
     int read_seq_from_file(const char* p, atom_Type_ID_t & model);   // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
     void make_seq_default() ;       // Make the sequence in a weired order
} ;


#endif
