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


# include paths
INCLUDES  := 


# Libraries
LIBRARIES := 
LDFLAGS   :=  


#LINKER
LINKOPT   :=

# GSL / MKL library
ifdef USE_MKL
     LDFLAGS   += -D_USE_MKL  # switch on flag for compiling

     MKLROOT   ?= ${INTELDIR}/mkl
     MKLLIB    ?= ${MKLROOT}/lib/intel64
     INCLUDES  += -I${MKLROOT}/include
     LIBRARIES += -L${MKLLIB}

     LDFLAGS   += -DMKL_ILP64
     LINKOPT   += -Wl,--no-as-needed   -lmkl_intel_lp64
     #LINKOPT   += -Wl,--start-group ${MKLLIB}/libmkl_intel_ilp64.a

     ifeq (${HOST_COMPILER}, icpc)
          ifdef USE_OPENMP
               LDFLAGS   +=  -qopenmp
               #LINKOPT   +=  ${MKLLIB}/libmkl_intel_thread.a ${MKLLIB}/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -mkl
               LINKOPT   +=   -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
          else
               #LINKOPT   +=  ${MKLLIB}/libmkl_sequential.a ${MKLLIB}/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -mkl
               LINKOPT   +=  -lmkl_sequential    -lmkl_core         -lpthread -lm -ldl
          endif
     else     
          LDFLAGS   += -m64
          ifdef USE_OPENMP
               #LINKOPT   +=  ${MKLLIB}/libmkl_gnu_thread.a ${MKLLIB}/libmkl_core.a -Wl,--end-group  -lgomp -lpthread -lm -ldl 
               LINKOPT   +=  -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 
          else
               #LINKOPT   +=  ${MKLLIB}/libmkl_sequential.a ${MKLLIB}/libmkl_core.a -Wl,--end-group  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl 
               LINKOPT   +=  -lmkl_sequential -lmkl_core            -lpthread -lm -ldl
          endif
     endif

else 

     # USE_GSL
     ifdef USE_GSL 
     LDFLAGS   += -D_USE_GSL  -lgslcblas
     endif

     # OPENMP
     ifdef USE_OPENMP
     LDFLAGS   += -fopenmp
     endif

endif


# NVCC flags
NVCCFLAGS := -Wno-deprecated-gpu-targets


#=====================================================================
#
# Target rules
all: clean build

build: gfunction_generate.x

DEP= tester.o Gfunction.o utility.o atomTypeID.o readGparams.o timestamps.o

gfunction_generate.x: $(DEP)
	$(HOST_COMPILER) $(INCLUDES) $(LIBRARIES) $(LINKOPT) $(CCFLAGS) $(LDFLAGS)  -o $@ $+


%.o: %.cpp 
	$(HOST_COMPILER) $(INCLUDES) $(LIBRARIES) $(CCFLAGS) $(LDFLAGS)  -o $@ -c $<


clean:
	rm -rf *o
	rm -f gfunction_generate.x

