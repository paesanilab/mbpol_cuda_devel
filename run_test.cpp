#include "twobodyForce.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <vector_functions.hpp>

int main() {

    boost::timer::auto_cpu_timer t;

    t.start();
    double *e_d;
    double3 *forces_d;
    double4 *posq_d, *periodicBoxSize_d;
    const unsigned int N_ATOMS = 6;
    size_t posq_size = N_ATOMS * sizeof(double4);
    size_t forces_size = N_ATOMS * sizeof(double3);
    double4 posq[N_ATOMS];
    double4 periodicBoxSize;

    double e[1];
    cudaMalloc((void **) &posq_d, posq_size);
    cudaMalloc((void **) &forces_d, forces_size);
    cudaMalloc((void **) &e_d, sizeof(double));
    cudaMalloc((void **) &periodicBoxSize_d, sizeof(double4));
    cudaMemcpy(posq_d, &posq, posq_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(periodicBoxSize_d, &periodicBoxSize, sizeof(double4), cudaMemcpyHostToDevice);
    t.stop();
    cudaMemset(e_d, -1234., sizeof(double));
    t.start();
    launch_evaluate_2b(posq_d, periodicBoxSize_d, forces_d, e_d);
    t.stop();
    t.report();
    t.start();
    cudaMemcpy(&e, e_d, sizeof(double), cudaMemcpyDeviceToHost);
    t.stop();
    std::cout << std::endl << "Energy: " << e[0] << " kcal/mol" << std::endl;
}
