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

PROGRAMS = compareAlgorithms compareMultiselect analyzeMultiselect realDataTests compareTopkselect
CompareAlgorithms = compareAlgorithms.cu bucketSelect.cu randomizedBucketSelect.cu
CompareMultiselect = compareMultiselect.cu generateProblems.cu bucketMultiselect.cu naiveBucketMultiselect.cu noExtremaRandomizedBucketSelect.cu multiselectTimingFunctions.cu
CompareTopkselect = compareTopkselect.cu randomizedTopkSelect.cu multiselectTimingFunctions.cu
AnalyzeMultiselect = analyzeMultiselect.cu bucketMultiselect.cu multiselectTimingFunctions.cu
RealDataTests = realDataTests.cu generateProblems.cu bucketMultiselect.cu 

all: $(PROGRAMS)

compareAlgorithms: $(addsuffix .o,$(CompareAlgorithms))
	$(LINK) -o compareAlgorithms  compareAlgorithms.cu.o $(NVCCFLAGS)

compareMultiselect: $(addsuffix .o,$(CompareMultiselect))
	$(LINK) -o compareMultiselect  compareMultiselect.cu.o $(NVCCFLAGS)

compareTopkselect: $(addsuffix .o,$(CompareTopkselect))
	$(LINK) -o  compareTopkselect compareTopkselect.cu.o $(NVCCFLAGS)

analyzeMultiselect: $(addsuffix .o,$(AnalyzeMultiselect))
	$(LINK) -o  analyzeMultiselect analyzeMultiselect.cu.o $(NVCCFLAGS)

realDataTests: $(addsuffix .o,$(RealDataTests))
	$(LINK) -o realDataTests realDataTests.cu.o $(NVCCFLAGS)

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
