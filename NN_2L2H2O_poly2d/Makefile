# 
# Build Nueral Network tester from NVCC/GCC and libraries CUDNN/CUBLAS/CUDA
#
#
#============================================================================


# Location of the CUDA/CUDNN Toolkit
CUDA_PATH      ?= /usr/local/cuda
CUDNN_PATH     ?= $(CUDA_PATH)
HDF5_PATH      ?= /usr/local/hdf5

# architecture
HOST_ARCH      := $(shell uname -m)
TARGET_ARCH    ?= $(HOST_ARCH)


# compiler
HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# flags
CCFLAGS   := -std=c++11
NVCCFLAGS := -Wno-deprecated-gpu-targets
LDFLAGS   := -lcudnn -lcublas -lhdf5 -lhdf5_cpp -O2


# include paths
INCLUDES  := -I$(CUDA_PATH)/include
INCLUDES  += -I$(CUDNN_PATH)/include
INCLUDES  += -I$(HDF5_PATH)/include

LIBRARIES := -L$(CUDA_PATH)/lib64
LIBRARIES += -L$(CUDNN_PATH)/lib64
LIBRARIES += -L$(HDF5_PATH)/lib


#=====================================================================
#
# Target rules
all: clean build

build: NN_2L2H2O_poly2d 

%: %.cu 
	$(NVCC) $(INCLUDES) $(LIBRARIES) $(NVCCFLAGS) $(CCFLAGS) $(LDFLAGS) -o $@ $<
	
#%.o: %.hpp 
#	$(HOST_COMPILER) $(CCFLAGS) $(LDFLAGS) -o $@ -c $<
		
	
clean:
	rm -rf *o
	rm -f NN_2L2H2O_poly2d
	
