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
