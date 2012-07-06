CUDA_INSTALL_PATH ?= /usr/local/cuda

#nvcc -o compareAlgorithms compareAlgorithms.cu  -I./lib/ -I. -arch=sm_20 -lcurand -lm -lgsl -lgslcblas

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
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH) 
NVCCFLAGS += $(LIB_CUDA) 
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand -lm -lgsl -lgslcblas

default: compareAlgorithms compareMultiselect

compareAlgorithms: compareAlgorithms.cu bucketSelect.cu randomizedBucketSelect.cu
	$(NVCC) -o compareAlgorithms compareAlgorithms.cu $(NVCCFLAGS)

#bucketSelect: bucketSelect.cu
#	$(NVCC) -o bucketSelect bucketSelect.cu $(NVCCFLAGS)

compareMultiselect: compareMultiselect.cu bucketMultiselect.cu naiveBucketMultiselect.cu
	$(NVCC) -o compareMultiselect compareMultiselect.cu $(NVCCFLAGS)

clean:
	rm -f compareAlgorithms compareMultiselect *~ x
