#ifndef TWOBODYFORCE
#define TWOBODYFORCE

#include <vector_functions.hpp>

void launch_evaluate_2b(
        const double4* __restrict__ posq,
        const double4* periodicBoxSize,
        double3 * forces,
        double * energy);

#endif
