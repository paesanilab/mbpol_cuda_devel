MBPol CUDA development repository
===========================

This repository contains a standalone code to evaluate two body interactions on CUDA.

It can be customized through `CMakeLists.txt` to either use the CUDA kernel `twobodyForceNN.cu` which for now is just empty or use `twobodyForce.cu` that contains the `MBPol` implementation of the polynomials.

This works as a starting point for working on the Neural Nets implementation following the Python version available on 
Github at <https://github.com/paesanilab/NeuralNets/tree/master/testcase_forCUDA>.

We will need to create the Neural Net architecture with CuDNN, load the weights already trained in Python and then evaluate the network to compute the Energy of the two body interaction.

This wrapper already defines the positions of the atoms of 2 water molecules and copies them to the device.
Check `twobodyForce.cu` for an example of how the data can be used from CUDA.

## Requirements

(already installed on `daisy`)

* CUDA >= 7
* Boost
* CMake

## Compilation

* Checkout this repository
* (optional) edit `CMakeLists.txt` to choose if you want to build Polynomials or Neural Nets (empty for now)
* create a folder named `build/` **inside** the repository folder
* enter the `build/` folder
* run CMake with:

        cmake ..

* run `make`
* execute the code with `./run_test`
