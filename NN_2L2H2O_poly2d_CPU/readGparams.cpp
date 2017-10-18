#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <algorithm>

#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"




using namespace std;


template<>
double Gparams_t<double>::return_a_number(string _string){
     return  stod ( _string );     
}

template<>
float Gparams_t<float>::return_a_number(string _string){
     return  stof ( _string );     
}


/*
template<typename T>
int Gparams_t<T>::read_param_from_file(const char* p, atom_Type_ID_t & model){
    try {
          ifstream ifs(p);
          string line;       
                   
          while(getline(ifs, line)){      // get every line as in-file-stream
               istringstream iss(line);   // get every line as in-string-stream 
               vector<string> record;
               
               // split the records in line by space
               while(iss){
                    string next;
                    if (!getline(iss,next, ' ') ) break;
                    if (next != ""){
                         record.push_back(next);
                    }
                  
               }
               
               vector<T> currnumbers;   // saved numbers in current line
               string atom_type = "";        // the first atom in the line is the main atom 
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




template<typename T>
int Gparams_t<T>::read_seq_from_file(const char* _file, atom_Type_ID_t & model) {
     try {
          ifstream ifs(_file);
          string line;       
          
          
          while(getline(ifs, line)){      // get every line as in-file-stream
               
               // trimming leading space
               line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));      
               
                        
               if ( line.size() > 0 && line[0] != COMMENT_IN_SEQ ) {    //  if start with `COMMENT_IN_SEQ`, it is a comment ; if length=0, it is a blank line

                    stringstream iss(line);   // get the line as in-string-stream 
                    vector<string> record;
               
                    // split the records in line by space
                    while(iss){
                         string next;
                         if (!getline(iss,next, ' ') ) break;
                         if (next != ""){
                              record.push_back(next);
                         }                  
                    }
               
                    
                    string atom_type = "";        // the first atom in the line is the main atom 
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

template<typename T>
void Gparams_t<T>::make_seq_default(){     
     for(auto it=params.begin(); it!=params.end(); it++){          
          for(auto it2=it->second.begin(); it2!=it->second.end(); it2++){
               seq[it->first].push_back(it2->first);           
          }
     }
}
*/


//===============================================================================
//
// template realization
void _NEVER_USED_INSTANLIZATION_READGPARAMS(){
     Gparams_t<float> g1;
     Gparams_t<double> g2;     
}



//===============================================================================
// Tester
/*
int main(int argc, char** argv) {
    
     Gparams_t Gparams;
     
     atom_Type_ID_t model;
     
     Gparams.readfromfile(argv[1], model);
     
     
     cout << Gparams.params["H"][18][6][2] << endl;
     
     cout << "Hello Hello" << endl;
     
     return 0;
}
*/
