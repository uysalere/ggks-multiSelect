SHELL = /bin/sh
CUDA_INSTALL_PATH ?= /usr/local/cuda

CPP := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc -ccbin /usr/bin
.SUFFIXES: .c .cpp .cu .o

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ 
# Libraries
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand -lm -lgsl -lgslcblas
# ARCH
ARCH = -arch=sm_20

# Common flags
 COMMONFLAGS += $(INCLUDES)
# Compilers
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH)
NVCCFLAGS += $(LIB_CUDA)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

PROGRAMS = \
compareAlgorithms \
compareMultiselect \
analyzeMultiselect \
realDataTests \
compareTopkselect

CompareAlgorithms = \
compareAlgorithms.cu \
bucketSelect.cu randomizedBucketSelect.cu noExtremaRandomizedBucketSelect.cu \
generateProblems.cu timingFunctions.cu

CompareMultiselect = \
compareMultiselect.cu \
bucketMultiselect.cu naiveBucketMultiselect.cu \
generateProblems.cu multiselectTimingFunctions.cu

CompareTopkselect = \
compareTopkselect.cu \
randomizedTopkSelect.cu \
generateProblems.cu multiselectTimingFunctions.cu

AnalyzeMultiselect = \
analyzeMultiselect.cu \
bucketMultiselect.cu \
multiselectTimingFunctions.cu

RealDataTests = \
realDataTests.cu \
bucketMultiselect.cu \
generateProblems.cu 

all: $(PROGRAMS)

compareAlgorithms: $(CompareAlgorithms)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

compareMultiselect: $(CompareMultiselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

compareTopkselect: $(CompareTopkselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

analyzeMultiselect: $(AnalyzeMultiselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

realDataTests: $(RealDataTests)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(PROGRAMS) *~ *.o

#compareAlgorithms: compareAlgorithms.cu bucketSelect.cu randomizedBucketSelect.cu
#	$(NVCC) -o compareAlgorithms compareAlgorithms.cu $(NVCCFLAGS)
