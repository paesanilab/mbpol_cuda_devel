#ifndef ATOMTYPEID_H
#define ATOMTYPEID_H

#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <vector>

typedef unsigned int idx_t;

extern const idx_t DEFAULT_ID;


// Class saving the atom type and atom ID in the model
class atom_Type_ID_t{
private:
     std::queue<idx_t> RESERVED_TYPE_INDEX;           // Type index list (use prime number 2,3,5 ... , explained in .cpp file)
     idx_t RESERVED_ATOM_INDEX;                       // Atom index (0, 1, 2, )

public:
    
     // Struct containing one type information
     struct type{
          std::string name;                       // Type name
          idx_t id;                               // Type index
          std::vector<idx_t> atom_list;           // Which atoms belong to this type
          
          type();
          type(std::string name, idx_t id);
          ~type();
          type(const type& that);
          
          bool insert_atom(idx_t _atomid);        // Helper function. Add atom id to this type `atom_list`
     };
     
     // Struct containing one atom information
     struct atom{
          std::string name;
          std::string type;
          idx_t id;                         
          
          atom();
          atom(std::string name, std::string type, idx_t);
          ~atom();
          atom(const atom& that);
     };


     // constructor/destructor/copy-constructor     
     atom_Type_ID_t();
     ~atom_Type_ID_t();
     atom_Type_ID_t(const atom_Type_ID_t& that);
     
     // Maps saving all atom and type information
     std::map<idx_t, atom*> atoms;
     std::map<std::string, type*> types;
     
     // insert a type / an atom
     idx_t insert_type(std::string _type);
     idx_t insert_atom(std::string _atom_name, std::string _type_name);
     
     // query an atom
     idx_t find_atom_idx_by_name(std::string _atom_name);

          
     //=========================================================================
     // Following functions are only used for water dimer mbpol NN model
     //=========================================================================
     
     // loading a default setting for mbpol NN model 
     void load_default_atom_id(double**& _idxary, size_t & _size);
     
     
     
     

};

#endif
