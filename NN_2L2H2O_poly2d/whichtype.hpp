#if !defined(_WHICHTYPE_H_)
#define _WHICHTYPE_H_


// Following four structs test the type of data 
template <typename T>
struct TypeIsFloat
{
     static const bool value = false;
};

template <>
struct TypeIsFloat<float>
{    
     static const bool value = true;
};


template <typename T>
struct TypeIsDouble
{
     static const bool value = false;
};

template <>
struct TypeIsDouble<double>
{    
     static const bool value = true;
};

#endif
