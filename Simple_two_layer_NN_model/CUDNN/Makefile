# 
# Build Nueral Network tester from NVCC/GCC and libraries CUDNN/CUBLAS/CUDA
#
#
#============================================================================


# Location of the CUDA/CUDNN Toolkit
CUDA_PATH      ?= /usr/local/cuda
CUDNN_PATH     ?= $(CUDA_PATH)

# architecture
HOST_ARCH      := $(shell uname -m)
TARGET_ARCH    ?= $(HOST_ARCH)


# compiler
HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# flags
CCFLAGS   := -std=c++11
NVCCFLAGS := -Wno-deprecated-gpu-targets
LDFLAGS   := -lcudnn -lcublas


# include paths
INCLUDES  := -I$(CUDA_PATH)/include
INCLUDES  += -I${CUDNN_PATH}/include
LIBRARIES := -L$(CUDA_PATH)/lib64
LIBRARIES += -L$(CUDNN_PATH)/lib64

#=====================================================================
#
# Target rules
all: clean build run

build: two_layer_NN

OBJ= two_layer_NN.o
DEP= error_util.hpp network.cu layer_toy_param.dat

two_layer_NN: $(OBJ) 
	$(NVCC) $(NVCCFLAG) $(LIBRARIES) $(LDFLAGS) -o $@ $+

%.o: %.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(CCFLAGS) -o $@ -c $<
	
%.o: %.hpp 
	$(HOST_COMPILER) $(CCFLAGS) -o $@ -c $<
		
	
clean:
	rm -rf *o
	rm -rf two_layer_NN
	
run:	
	./two_layer_NN	
