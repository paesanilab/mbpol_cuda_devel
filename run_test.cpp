#include "twobodyForce.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <vector_functions.hpp>

int main() {

        boost::timer::auto_cpu_timer t;

        t.start();

        const unsigned int N_ATOMS = 6;
        // Define host variables
        double4 posq[N_ATOMS]; // using OpenMM convention of position array also including charge q, so it is x,y,z pos and charge
        double e[1];

        // Define positions from the unit test at:
        // https://github.com/paesanilab/mbpol_openmm_plugin/blob/master/platforms/cuda/tests/TestCudaMBPolTwoBodyForce.cpp#L60

        posq[0].x = -1.516074336e+00; //[A]
        posq[0].y = -2.023167650e-01;
        posq[0].z = 1.454672917e+00;
        posq[1].x = -6.218989773e-01;
        posq[1].y = -6.009430735e-01;
        posq[1].z = 1.572437625e+00;
        posq[2].x = -2.017613812e+00;
        posq[2].y = -4.190350349e-01;
        posq[2].z = 2.239642849e+00;
        posq[3].x = -1.763651687e+00;
        posq[3].y = -3.816594649e-01;
        posq[3].z = -1.300353949e+00;
        posq[4].x = -1.903851736e+00;
        posq[4].y = -4.935677617e-01;
        posq[4].z = -3.457810126e-01;
        posq[5].x = -2.527904158e+00;
        posq[5].y = -7.613550077e-01;
        posq[5].z = -1.733803676e+00;

        // Define device pointers
        double *e_d;
        double3 *forces_d;
        double4 *posq_d;

        // Allocate memory on the device
        size_t posq_size = N_ATOMS * sizeof(double4);
        size_t forces_size = N_ATOMS * sizeof(double3);
        cudaMalloc((void **) &posq_d, posq_size);
        cudaMalloc((void **) &forces_d, forces_size);
        cudaMalloc((void **) &e_d, sizeof(double));
        cudaMemcpy(posq_d, &posq, posq_size, cudaMemcpyHostToDevice);
        // cudaMemset(e_d, -1234., sizeof(double)) // easy way to check if energy is being written by the kernel;
        
        t.stop();
        std::cout << std::endl << "Setup completed" << std::endl;
        t.report();

        // Launch the CUDA kernel to compute energy and forces
        std::cout << std::endl << "Launch the CUDA kernel" << std::endl;
        t.start();
        launch_evaluate_2b(posq_d, forces_d, e_d);
        t.stop();
        t.report();

        // Copy results back to the host
        double expectedEnergy = 6.14207815;
        cudaMemcpy(&e, e_d, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << std::endl << "Energy: " << e[0] << " kcal/mol" << std::endl;
        std::cout << "Expected Energy: " << expectedEnergy << " kcal/mol" << std::endl;
}
