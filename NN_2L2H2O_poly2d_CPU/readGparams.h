#ifndef GPARAMS_H
#define GPARAMS_H

#include <vector>
#include <string>
#include <map>
#include <algorithm>

#include "atomTypeID.h"
#include "utility.h"


// The comment line in "seq.file" starts with following symbol
#define COMMENT_IN_SEQ '#'

// Read parameters for G-function construction
template <typename T>
struct Gparams_t {
public:
     std::map<std::string,  std::map<idx_t, std::vector<std::vector<T> > > > params;  
     
     std::map<std::string,  std::vector<idx_t> > seq;                   
     
     
     // Read in params from a file; save numbers in `params`; save related atom information into `model`.
     int read_param_from_file(const char* p, atom_Type_ID_t & model){
         try {
               std::ifstream ifs(p);
               std::string line;       
                        
               while(getline(ifs, line)){      // get every line as in-file-stream
                    std::istringstream iss(line);   // get every line as in-string-stream 
                    std::vector<std::string> record;
                    
                    // split the records in line by space
                    while(iss){
                         std::string next;
                         if (!getline(iss,next, ' ') ) break;
                         if (next != ""){
                              record.push_back(next);
                         }
                       
                    }
                    
                    std::vector<T> currnumbers;   // saved numbers in current line
                    std::string atom_type = "";        // the first atom in the line is the main atom 
                    idx_t atom_relation =1;       // the relationship index ( = multiple of atom type index other than the main atom )
                    
                    // for every line: 
                    //        - find param's atom type by first string
                    //        - find all other atoms types, and generate a unique matrix mapping index by the multiplication of atom type index
                    //        - save all numbers (double/single precision) as vector of vector and map it by atom type + matrix mapping index 
                    for(auto it=record.begin(); it!=record.end(); it++){                              
                         if ( IsFloat<T>(*it)){
                              T f = return_a_number(*it);               
                              currnumbers.push_back(f);               
                         } else {                        
                              if(atom_type == "")  atom_type = *it;
                              
                              auto it2 = model.types.find(*it) ;                          
                              idx_t curridx =1; 
                              
                              if( it2 == model.types.end() ){  
                                   curridx = model.insert_type(*it); 
                              }else {
                                   curridx = model.types[*it]->id;
                              }
                              
                              atom_relation *= curridx;               
                         }
                    }
                                                                                                                                                            
                    if ( currnumbers.size() >0 && atom_relation>1 ){   
                         params[atom_type][atom_relation].push_back(currnumbers);            
                    }        
               }          
               return 0;
         } catch (const std::exception& e) {
             std::cerr << " ** Error ** : " << e.what() << std::endl;
             return 1;
         }    ;
     };



     // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
     int read_seq_from_file(const char* _file, atom_Type_ID_t & model) {
          try {
               std::ifstream ifs(_file);
               std::string line;       
               
               
               while(getline(ifs, line)){      // get every line as in-file-stream
                    
                    // trimming leading space
                    line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));      
                    
                             
                    if ( line.size() > 0 && line[0] != COMMENT_IN_SEQ ) {    //  if start with `COMMENT_IN_SEQ`, it is a comment ; if length=0, it is a blank line

                         std::stringstream iss(line);   // get the line as in-string-stream 
                         std::vector<std::string> record;
                    
                         // split the records in line by space
                         while(iss){
                              std::string next;
                              if (!getline(iss,next, ' ') ) break;
                              if (next != ""){
                                   record.push_back(next);
                              }                  
                         }
                    
                         
                         std::string atom_type = "";        // the first atom in the line is the main atom 
                         idx_t atom_relation =1;       // the relationship index ( = multiple of atom type index other than the main atom )

                         // for every line: 
                         //        - find base atom type by first string
                         //        - find all other atoms types, and generate a unique matrix mapping index by the multiplication of atom type index
                         //        - save all numbers (double precision) as vector of vector and map it by atom type + matrix mapping index 
                         for(auto it=record.begin(); it!=record.end(); it++){                              
                                      
                                   if(atom_type == "")  atom_type = *it;

                                   auto it2 = model.types.find(*it) ;                          
                                   idx_t curridx =1; 
                                   if( it2 == model.types.end() ){  
                                        curridx = model.insert_type(*it); 
                                   }else {
                                        curridx = model.types[*it]->id;
                                   }                             
                                   atom_relation *= curridx;               
                         }                                                                                                                                                       
                         seq[atom_type].push_back(atom_relation);
                    }                                                  
               }          
               return 0;
          } catch (const std::exception& e) {
             std::cerr << " ** Error ** : " << e.what() << std::endl;
             return 1;
         }
     }

     // Make the sequence in a weired order
     void make_seq_default(){     
          for(auto it=params.begin(); it!=params.end(); it++){          
               for(auto it2=it->second.begin(); it2!=it->second.end(); it2++){
                    seq[it->first].push_back(it2->first);           
               }
          }
     }     
     
     
private:
     T return_a_number(std::string _string);   
     
     
     
       
} ;


#endif
