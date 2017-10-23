# 
# Build Nueral Network tester from NVCC/GCC and libraries CUDNN/CUBLAS/CUDA
#
#
#============================================================================


# Location of the CUDA/CUDNN Toolkit
CBLAS_PATH     ?= /usr/

# architecture
HOST_ARCH      := $(shell uname -m)
TARGET_ARCH    ?= $(HOST_ARCH)
 

# compiler
ifdef CMP_INTEL
HOST_COMPILER  = icpc
else ifdef CMP_GNU
HOST_COMPILER  = g++
else
HOST_COMPILER  = g++
endif

# Compiler flags
CCFLAGS   := -std=c++11 

ifdef ENABLE_XHOST
CCFLAGS   += -xHost
else ifdef ENABLE_CAVX512
CCFLAGS   += -xCOMMON-AVX512 -O3
else ifdef ENABLE_CRAVX512
CCFLAGS   += -xCORE-AVX512 -O3
else ifdef ENABLE_CRAVX2
CCFLAGS   += -xCORE-AVX2 -O3
else ifdef ENABLE_CRAVXI
CCFLAGS   += -xCORE-AVX-I -O3
else ifdef ENABLE_AVX
CCFLAGS   += -xAVX -O3
else 
CCFLAGS   += -O3
endif


# Libraries
LDFLAGS   :=  

# GSL / MKL library
ifdef USE_MKL
LDFLAGS   += -D_USE_MKL
else ifdef USE_GSL 
LDFLAGS   += -D_USE_GSL  -lgslcblas
else
#LDFLAGS   += -D_USE_GSL
endif


# OPENMP
ifdef USE_OPENMP
LDFLAGS   += -fopenmp
endif


# NVCC flags
NVCCFLAGS := -Wno-deprecated-gpu-targets


# include paths
#INCLUDES  := -I$(CBLAS_PATH)/include
#LIBRARIES := -L$(LD_LIBRARY_PATH)
#LIBRARIES := -L$(CBLAS_PATH)/lib64
#LIBRARIES  := -L/usr/lib64

#=====================================================================
#
# Target rules
all: clean build

build: gfunction_generate.x

DEP= tester.o Gfunction.o utility.o atomTypeID.o readGparams.o timestamps.o

gfunction_generate.x: $(DEP)
	$(HOST_COMPILER) $(CCFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBRARIES) -o $@ $+

	
%.o: %.cpp 
	$(HOST_COMPILER) $(CCFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBRARIES) -o $@ -c $<
		
	
clean:
	rm -rf *o
	rm -f gfunction_generate.x
	
