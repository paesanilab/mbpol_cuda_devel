extern "C" __device__ double computeInteraction(
        const unsigned int atom1,
        const unsigned int atom2,
        const double4* __restrict__ posq,
        const double4* periodicBoxSize,
        double3 * forces) {

        // CUDA COMPUTATIONAL KERNEL

        return 0;
}

__global__ void evaluate_2b(
        const double4* __restrict__ posq,
        const double4* periodicBoxSize,
        double3 * forces,
        double * energy) {
        // This function will parallelize computeInteraction to run in parallel with all the pairs of molecules,
        // for now just computing the interaction between molecule 0 and molecule 1
        energy[0] = computeInteraction(0, 3, posq, periodicBoxSize, forces);
}

void launch_evaluate_2b(
        const double4* __restrict__ posq,
        const double4* periodicBoxSize,
        double3 * forces,
        double * energy) {
        // This function is useful to configure the number of device threads, just one for now:
        evaluate_2b<<<1,1>>>(posq, periodicBoxSize, forces, energy);
        cudaDeviceSynchronize();
}
