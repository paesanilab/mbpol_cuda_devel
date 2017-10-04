#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <limits>
#include "atomTypeID.h"
#include "utility.h"

using namespace std;

const idx_t DEFAULT_ID = std::numeric_limits<idx_t>::max();


//===================================================================
// ATOM and TYPE struct in atom_TYPE_ID_t
// Constructor/destructor/copy-constructor
atom_Type_ID_t::atom::atom(){
     name="";
     type="";
     id=DEFAULT_ID;
};


atom_Type_ID_t::atom::atom(string _name, string _type, idx_t _id){
     name=_name;
     type=_type;
     id=_id;
};

atom_Type_ID_t::atom::~atom(){};

atom_Type_ID_t::atom::atom(const atom_Type_ID_t::atom & that){
     name=that.name;
     type=that.type;
     id=that.id;
};


atom_Type_ID_t::type::type(){
     name="";
     id=DEFAULT_ID;
     atom_list.clear();
};


atom_Type_ID_t::type::type(string _name, idx_t _id){
     name=_name;
     id=_id;
     atom_list.clear();
};

atom_Type_ID_t::type::~type(){};

atom_Type_ID_t::type::type(const atom_Type_ID_t::type &that){
     name=that.name;
     id=that.id;
     atom_list=that.atom_list;
};


// helper function inserting an atom to this type 
bool atom_Type_ID_t::type::insert_atom(idx_t _atomid){
     for (auto it=atom_list.begin(); it!=atom_list.end(); it++){
          if (*it == _atomid ) return false;
     }
     atom_list.push_back(_atomid);
     return true; 
};


//=========================================================================
// atom_TYPE_ID_t definition
// constructor/destructor/copy-constructor
atom_Type_ID_t::atom_Type_ID_t(){
     // prepare a prime number list for each atom type      
     // Reason: in the simulation, one needs to find the relationship between one atom and other several atoms.
     // This indicates a combination of their types: e.g. for a Hydrogen atom "H1", 
     // one needs to analyze the influence from another Hydrogen atom "H2" and an oxygen atom "O1",
     // so that the relationship of Hydrogen-Hydrogen-Oxygen (H1-H2-O1) is needed.
     // Note however, the sequence of atom type in the relationship (other than the first one) should not matter: 
     // both H1-H2-O1 and H1-O1-H2 give the same information.
     // To index the exchangable sequence of the atom types, prime number is used as atom type index,
     // so that the multiple of type indexes indicates a unique type combination 
     // without depending on the sequence of atom types:
     // Let H_index=2 and O_index=3, so that: 
     //        HO = H_index*O_index = O_index*H_index = 6
     //        HH = H_index*H_index = 4
     //        OO = O_index*O_index = 9
     idx_t LIST[25] = {                      // default loading 25 atom types
          2 , 3, 5, 7 , 11 , 13, 17, 19,
          23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79,
          83, 89, 97            
     };            
     for(int i =0; i<25; i++){
          RESERVED_TYPE_INDEX.push(LIST[i]);
     };
     RESERVED_ATOM_INDEX=0;
};
atom_Type_ID_t::~atom_Type_ID_t(){
     for(auto it=atoms.begin(); it!=atoms.end(); it++){
          delete it->second;
          it->second = nullptr;
     };
     for(auto it=types.begin(); it!=types.end(); it++){
          delete it->second;
          it->second = nullptr;
     };              
};
atom_Type_ID_t::atom_Type_ID_t(const atom_Type_ID_t & that){
     RESERVED_TYPE_INDEX = that.RESERVED_TYPE_INDEX;
     RESERVED_ATOM_INDEX = that.RESERVED_ATOM_INDEX;
     
     for(auto it=that.atoms.begin(); it!=that.atoms.end();it++){
          atom_Type_ID_t::atom *_newatom = new atom_Type_ID_t::atom(it->second->name, it->second->type, it->second->id);
          atoms.insert(map<idx_t, atom_Type_ID_t::atom*>::value_type(it->second->id, _newatom));     
     }     
     for(auto it=that.types.begin(); it!=that.types.end();it++){
          atom_Type_ID_t::type *_newtype = new atom_Type_ID_t::type(it->second->name, it->second->id);
          _newtype->atom_list = it->second->atom_list;
          types.insert(map<string, atom_Type_ID_t::type*>::value_type(it->second->name, _newtype));     
     };     
};         



// Insert new type or atom 
// Try inserting a type:
//   - if successes, return new type index;
//   - if fails, retrieve the type index and return;
idx_t atom_Type_ID_t::insert_type(string _type){
     auto it = types.find(_type);
     if ( ( it != types.end()) && (it->second->id != DEFAULT_ID ) ) {
          // if found the type, return its id;
          return it->second->id;
     } else {
          // not found, create a new type
          idx_t _id = RESERVED_TYPE_INDEX.front();
          RESERVED_TYPE_INDEX.pop();          
          atom_Type_ID_t::type *new_type = new atom_Type_ID_t::type(_type, _id);          
          types[_type] = new_type;
          return _id;
     };
};

// Try inserting an atom and return its atom index;
idx_t atom_Type_ID_t::insert_atom(string _name, string _type){
     for (auto it = atoms.begin(); it!=atoms.end(); it++){     
          // find via all atoms;
          if (it->second->name == _name ) {
               return it->second->id;
          };                    
     };     
     // Not existing atom, insert it!
     idx_t _id = RESERVED_ATOM_INDEX++;
     atom_Type_ID_t::atom * _new_atom = new atom_Type_ID_t::atom(_name, _type, _id);     
     atoms.insert(map<idx_t, atom_Type_ID_t::atom*>::value_type(_id, _new_atom));
     idx_t type_id = insert_type(_type); // try inserting as a new type
     return _id;
};


// find an atom ID by its name
idx_t atom_Type_ID_t::find_atom_idx_by_name(std::string _name){    
    for (auto it = atoms.begin(); it!=atoms.end(); it++){     
          // find via all atoms;
       if (it->second->name == _name ) {
          return it->second->id;
       };                    
    };        
    return DEFAULT_ID;
};


// an default atom name and type loader
// Array _idxary[atom1][atom2] contains the column index in distance matrix
//   between [atom1] and [atom2].
// So that _idxary[atom1][atom2] = _idxary[atom2][atom1]
void atom_Type_ID_t::load_default_atom_id(double**& _idxary, size_t & _size){
     insert_atom("H1(a)", "H");
     insert_atom("H2(a)", "H");
     insert_atom("H1(b)", "H");
     insert_atom("H2(b)", "H");
     insert_atom("O(a)", "O");
     insert_atom("O(b)", "O");
     
     
     // The sequence of the pairs in the vector indicates the column index 
     // in distance matrix array
     vector< pair<string, string> > mappings;
     mappings.push_back(std::make_pair("H1(a)","H2(a)"));
     mappings.push_back(std::make_pair("H1(b)","H2(b)"));
     mappings.push_back(std::make_pair("O(a)","H1(a)"));
     mappings.push_back(std::make_pair("O(a)","H2(a)"));
     mappings.push_back(std::make_pair("O(b)","H1(b)"));
     mappings.push_back(std::make_pair("O(b)","H2(b)"));
     mappings.push_back(std::make_pair("H1(a)","H1(b)"));
     mappings.push_back(std::make_pair("H1(a)","H2(b)"));
     mappings.push_back(std::make_pair("H2(a)","H1(b)"));
     mappings.push_back(std::make_pair("H2(a)","H2(b)"));
     mappings.push_back(std::make_pair("O(a)","H1(b)"));
     mappings.push_back(std::make_pair("O(a)","H2(b)"));
     mappings.push_back(std::make_pair("O(b)","H1(a)"));
     mappings.push_back(std::make_pair("O(b)","H2(a)"));
     mappings.push_back(std::make_pair("O(a)","O(b)"));
     
     
     _size = atoms.size();

     init_mtx_in_mem(_idxary, _size, _size);
     
     
     idx_t idx_first, idx_second;
     idx_t idx=0;
     for(auto it=mappings.begin(); it!=mappings.end(); it++){
          idx_first = find_atom_idx_by_name(it->first);
          idx_second= find_atom_idx_by_name(it->second);
          _idxary[idx_first][idx_second]=idx;
          _idxary[idx_second][idx_first]=idx;
          idx++;
     };          

};


// a tester
/*
int main(void){
     
     
     atom_Type_ID_t Model;
     
     double** colidx;
     size_t natom;
     
     Model.load_default_atom_id(colidx, natom);
     
     Model.insert_atom("HOOH", "H");
     Model.insert_type("Cl");
          
     cout << Model.atoms[0]->name << " , " << Model.atoms[0]->type << endl;
     cout << Model.types["Cl"]->id << endl; 
     
     
     atom_Type_ID_t::type * _new_type = new atom_Type_ID_t::type("Cl", 112);
     
     
     delete Model.types["Cl"] ;
     Model.types["Cl"] = _new_type;
        
     
     cout << Model.find_atom_idx_by_name("HOOH") << endl;
     
     cout << Model.types.size() << endl;
     
     
     cout << colidx[5][3] << endl;
     
     //delete _new_type;
     delete[] colidx[0];
     delete[] colidx;
     
     return 0;
};
*/

