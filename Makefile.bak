#nvcc -o compareAlgorithms compareAlgorithms.cu  -I./lib/ -I. -arch=sm_20 -lcurand -lm -lgsl -lgslcblas
CUDA_INSTALL_PATH ?= /usr/local/cuda

CPP := g++
CC := gcc
NVCC := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ 
# ARCH
ARCH = -arch=sm_20
# Libraries
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand -lm -lgsl -lgslcblas
# Common flags
COMMONFLAGS += $(INCLUDES)
# Compilers
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH) 
NVCCFLAGS += $(LIB_CUDA) 
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

default: compareAlgorithms compareMultiselect analyzeMultiselect realDataTests

compareAlgorithms: compareAlgorithms.cu bucketSelect.cu randomizedBucketSelect.cu
	$(NVCC) -o compareAlgorithms compareAlgorithms.cu $(NVCCFLAGS)

compareMultiselect: compareMultiselect.cu bucketMultiselect.cu naiveBucketMultiselect.cu generateProblems.cu noExtremaRandomizedBucketSelect.cu multiselectTimingFunctions.cu
	$(NVCC) -o compareMultiselect compareMultiselect.cu $(NVCCFLAGS)

compareTopkselect: compareTopkselect.cu compareTopkselect.hpp multiselectTimingFunctions.cu randomizedTopkSelect.cu
	$(NVCC) -o compareTopkselect compareTopkselect.cu $(NVCCFLAGS)

analyzeMultiselect: analyzeMultiselect.cu bucketMultiselect.cu multiselectTimingFunctions.cu
	$(NVCC) -o analyzeMultiselect analyzeMultiselect.cu $(NVCCFLAGS)

realDataTests: realDataTests.cu generateProblems.cu bucketMultiselect.cu 
	$(NVCC) -o realDataTests realDataTests.cu $(NVCCFLAGS)

clean:
	rm -f compareAlgorithms compareMultiselect compareTopkselect analyzeMultiselect realDataTests *~ x
