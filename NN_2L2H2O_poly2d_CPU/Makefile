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
HOST_COMPILER  ?= icpc

# flags
#CCFLAGS   := -std=c++11 -xCORE-AVX2 -O3 
#CCFLAGS   := -std=c++11 -xCOMMON-AVX512 -O3 
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


NVCCFLAGS := -Wno-deprecated-gpu-targets
LDFLAGS   := -lgslcblas -fopenmp 


# include paths
INCLUDES  := -I$(CBLAS_PATH)/include

LIBRARIES := -L$(LD_LIBRARY_PATH)
LIBRARIES += -L$(CBLAS_PATH)/lib64


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
	
