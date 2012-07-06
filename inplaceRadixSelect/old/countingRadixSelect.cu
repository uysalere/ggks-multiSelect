#include <thrust/scan.h>

namespace CountingRadixSelect{
 
template<typename T>
struct preProcess{
  __device__ __host__  void operator()(T &converted_key) {}
};
  
template<>
struct preProcess<float>{
  __device__ __host__  void operator()(unsigned int &converted_key) {
    unsigned int mask = (converted_key & 0x80000000) ? 0xffffffff : 0x80000000; 
    converted_key ^= mask;
  }
};
    
template<>
struct preProcess<double>{
  __device__ __host__  void operator()(unsigned long long &converted_key) {
    unsigned long long mask = (converted_key & 0x8000000000000000) ? 0xffffffffffffffff : 0x8000000000000000; 
    converted_key ^= mask;
  }
};


template<typename T>
void postProcess(uint *result ){}

template<>
void postProcess<float>(uint *result){
  unsigned int mask = (result[0] & 0x80000000) ?  0x80000000 : 0xffffffff ; 
  result[0] ^= mask; 
}

template<typename T>
void postProcess(unsigned long long *result){}

template<>
void postProcess<double>(unsigned long long *result){
  const unsigned long long mask = (result[0] & 0x8000000000000000) ? 0x8000000000000000 : 0xffffffffffffffff; 
  result[0] ^= mask;
}
 

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, T MASK0>
__global__ void getCounts(T *d_vec,const uint size,const T answerSoFar, uint *digitCounts,const T mask1,const uint offset,const T max){ 
    preProcess<T> functor;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i,j;
  T value;
  extern __shared__ ushort sharedCounts[];
  for(j = 0; j < 15;j++){
     sharedCounts[j * blockDim.x + threadIdx.x] = 0;
   }
  for(i = idx; i < size; i += offset){
    value = d_vec[i];
    functor(value);
    value &= MASK0;
    if(value >= answerSoFar && (value < max)){
       sharedCounts[((value >> BIT_SHIFT) & mask1) * blockDim.x + threadIdx.x]++;
    }
   }

   for(i = 0; i <mask1 ;i++){
     if(sharedCounts[blockDim.x * i + threadIdx.x]){
       digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
     }
   }
}

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, T MASK0>
__global__ void getCountsNotShared(const T *d_vec,const uint size,const T answerSoFar, uint *digitCounts,const T mask1,const uint offset,const T max){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;
  T value;
  for(i = idx; i < size; i += offset){
    value = d_vec[i] & MASK0;
    if(value >= answerSoFar && (value < max)){
      digitCounts[((value >> BIT_SHIFT) & mask1) * offset + idx]++;
    }
   }
}

template<typename T,uint BIT_SHIFT>
void updateAnswerSoFar(T &answerSoFar,T digitValue){
  T digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}
  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T determineDigit(uint *digitCounts, uint k, uint countSize,uint numThreads, uint size, uint &numSmaller){
  uint *numLessThanOrEqualToI;
  uint i=0, smaller = 0;
  uint adjustedSize = size - numSmaller;
  numLessThanOrEqualToI = (uint *) malloc(sizeof(uint));
  k = adjustedSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  thrust::inclusive_scan(ptr, ptr + (numThreads * (1 << RADIX_SIZE)) - 1, ptr);
  while(i < (1 << RADIX_SIZE) - 1){
    cudaMemcpy(numLessThanOrEqualToI, digitCounts + (i * numThreads + (numThreads - 1)), sizeof(uint), cudaMemcpyDeviceToHost);
    if(numLessThanOrEqualToI[0] >=  k){
      numSmaller += smaller;
      return i;
    }
    smaller = numLessThanOrEqualToI[0];
    i++;
  }

  numSmaller += smaller;
  return (1<< RADIX_SIZE) - 1;
}

  template<typename T>
  __global__ void partition(T d_vec, 
template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
void  digitPass(T *d_vec,uint size,uint k, T &answerSoFar, uint *digitCounts, uint blocks, uint threadsPerBlock, uint &numSmaller){
  T x = (T ((1 << RADIX_SIZE) - 1)) << BIT_SHIFT;
  //(((T 1)<< BIT_SHIFT) - 1)
    //  const T mask0 = ~((0x0000000000000001 << (BIT_SHIFT)) - 1);
  const T mask0 = ~( ((T) 1 << BIT_SHIFT) - 1);
  T  mask1 = (1 << (RADIX_SIZE)) - 1;
  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint countSize = (1 << RADIX_SIZE) * blocks * threadsPerBlock; 
  uint neededSharedMemory = ((1 << RADIX_SIZE) - 1) * threadsPerBlock * sizeof(ushort);
  cudaMemset(digitCounts,0, countSize * sizeof(uint));


  if(RADIX_SIZE <= 4){
    getCounts<T,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, size, answerSoFar, digitCounts,mask1,totalThreads, answerSoFar | x);
  }
  else{
    getCountsNotShared<T,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock>>>(d_vec, size, answerSoFar, digitCounts,mask1,totalThreads, answerSoFar | x);
  }

  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, size, numSmaller);
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
}


uint runDigitPasses(uint *d_vec, uint size, uint k,uint blocks, uint threadsPerBlock){
  uint  numSmaller = 0, answerSoFar = 0;
  uint *digitCounts;
  
  uint countSize = (1 << 8) * blocks * threadsPerBlock;
  cudaMalloc(&digitCounts, countSize * sizeof(uint));

  digitPass<uint,4,28>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<uint,4,24>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,20>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,16>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,12>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,8>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,4>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<uint,4,0>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);

  digitPass<uint,8,16>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<uint,8,8>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<uint,8,0>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);

  cudaFree(digitCounts);
  return answerSoFar;
}


 unsigned long long runDigitPasses( unsigned long long *d_vec, uint size, uint k,uint blocks, uint threadsPerBlock){
  uint  numSmaller = 0;
  unsigned long long answerSoFar = 0;
  uint *digitCounts;
  
  uint countSize = (1 << 4) * blocks * threadsPerBlock;
  cudaMalloc(&digitCounts, countSize * sizeof(uint));

  digitPass< unsigned long long,4,60>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,56>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,52>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,48>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,44>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,40>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,36>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,32>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,28>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,24>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,20>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,16>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,12>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,8>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,4>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass< unsigned long long,4,0>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);

  
  cudaFree(digitCounts);
  return answerSoFar;
}



template<typename T>
T countingRadixSelect(T *d_vec, uint size, uint k){
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  uint blocks =  dp.multiProcessorCount* 2;
  uint threadsPerBlock = dp.maxThreadsPerBlock / 2;
  return runDigitPasses(d_vec, size, k, blocks, threadsPerBlock);
}

}
