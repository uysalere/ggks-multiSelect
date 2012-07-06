#include <thrust/scan.h>
  #include <thrust/count.h>
namespace CountingRadixSelect{


  template<typename T,uint BIT_SHIFT>
 struct isLarger_x {
  isLarger_x(T x) : x(x) {}
    __device__ bool operator()(T y) { return ((y>>BIT_SHIFT) & 0xF) == x; }

   T x;
};

  template<typename T, uint BIT_SHIFT>
uint countEm(T *d_vec, uint size, uint digit){
    isLarger_x<T,BIT_SHIFT> isLargerFunctor(digit);
    thrust::device_ptr<T>d_ptr(d_vec);
    uint result = thrust::count_if(d_ptr, d_ptr + size, isLargerFunctor);
      return result;
  }
template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
__global__ void getCountsFirstPass(T *d_vec,const uint size, uint *digitCounts,const uint offset){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i,j;
  T value;
  extern __shared__ ushort sharedCounts[];
  for(j = 0; j < 16;j++){
     sharedCounts[j * blockDim.x + threadIdx.x] = 0;
   }
  for(i = idx; i < size; i += offset){
    value = d_vec[i];
    sharedCounts[(value >> BIT_SHIFT) * blockDim.x + threadIdx.x]++;
   }

   for(i = 0; i <16 ;i++){
       digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
   }
}

template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, T MASK0>
  __global__ void getCounts(T *d_vec,T *reduced_vec, const uint size,const T answerSoFar, uint *digitCounts, uint *altDigitCounts,
                            const T mask1,const uint offset,const T max, const uint previousDigit, uint numSmaller ){ 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i,j;
    uint startCount = altDigitCounts[ previousDigit * offset + idx] - numSmaller;
    T value;
    extern __shared__ ushort sharedCounts[];
    reduced_vec[startCount] = 15;

    for(j = 0; j < 16;j++){
      sharedCounts[j * blockDim.x + threadIdx.x] = 0;
    }
    for(i = idx; i < size; i += offset){
      value = d_vec[i];
    
      if(((value >> (BIT_SHIFT + 4)) & 0xF) ==previousDigit){
        sharedCounts[((value >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;
      
          reduced_vec[startCount] = 0;
          reduced_vec[startCount] = value;
          startCount += 1;
       }
    }
    for(i = 0; i < 16 ;i++){
      digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
    }
  }


template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, T MASK0>
__global__ void partition(uint * d_vec, uint *alt_vec, uint size, uint *digitCounts, uint offset, uint digit, uint numSmaller){
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       uint startIndex = digitCounts[ digit * offset + idx] - numSmaller;
       int i;
       T value;
       for(i = idx; i < size; i += offset){
         value = d_vec[i];
         if(((value >> BIT_SHIFT) & 0xF) == digit){
           alt_vec[startIndex++] = value;
         }
       }
}


template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
uint determineDigit(uint *digitCounts, uint &k, uint countSize,uint numThreads, uint &currentSize,uint &nextSize, uint &numSmaller){
  uint *numLessThanOrEqualTo, *nlt2;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  nlt2 = (uint *) malloc(16 * sizeof(uint));

  uint i=0;
  uint *ns;
  ns = (uint *)malloc(sizeof(uint));
  currentSize = nextSize;
  uint threshold = nextSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  uint total =0; uint temp;
  thrust::exclusive_scan(ptr, ptr + (numThreads * 16)+1, ptr);
  printf("VECTOR Of Offsets2\n");
  PrintFunctions::printCudaArray(digitCounts +countSize-1 , 1);
  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }
    for(i =0; i < 16; i++){
    cudaMemcpy(nlt2 + i, digitCounts + ((i+1) * numThreads) + 1, sizeof(uint), cudaMemcpyDeviceToHost);
  }
 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    k -= nextSize - numLessThanOrEqualTo[0];
    nextSize = numLessThanOrEqualTo[0];
    numSmaller = 0;
  }
  printf("%d  %d %d\n", 0 , numLessThanOrEqualTo[0], nlt2[0]);

  for(i = 1; i < 16; i++){
    printf("%d %d  %d\n", i , numLessThanOrEqualTo[i], nlt2[i]);
    if(numLessThanOrEqualTo[i] >= threshold){
      k -= nextSize - numLessThanOrEqualTo[i];
      cudaMemcpy(ns, digitCounts + ((i-1) * numThreads + numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
      nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      numSmaller= ns[0];
      return i;
    }
  }
  printf("OPPS\n");
  return 5367;
}


template<typename T,uint BIT_SHIFT>
void updateAnswerSoFar(T &answerSoFar,T digitValue){
  T digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}

template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T  digitPass(uint *d_vec, uint *alt_vec, uint &currentSize, uint &nextSize, uint k, uint answerSoFar, uint *digitCounts, uint *altDigitCounts,uint blocks,uint threadsPerBlock,uint countSize, uint &numSmaller, uint previousDigit){
  T x = (T ((1 << RADIX_SIZE) - 1)) << BIT_SHIFT;
  const T mask0 = ~( ((T) 1 << BIT_SHIFT) - 1);
  T  mask1 = (1 << (RADIX_SIZE)) - 1;
  printf("\n\n\nBIT_SHIFT: %d\nCSIZE:%d\nNSIZE:%d\nK:%d\nPREVIOUS:%d\nNUM SMALLER:%d\n",BIT_SHIFT,currentSize,nextSize,k, previousDigit,numSmaller);


  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * threadsPerBlock * sizeof(ushort);
  cudaMemset(digitCounts,0, countSize * sizeof(uint));

  //if(BIT_SHIFT == 28){
   getCountsFirstPass<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, currentSize,digitCounts,totalThreads);
   // }
 //  else{
 
 // getCounts<T,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock,neededSharedMemory>>>(d_vec,alt_vec, currentSize, answerSoFar, digitCounts,
 //                                                                                           altDigitCounts, mask1,totalThreads,
 //                                                                                           answerSoFar | x, previousDigit, numSmaller);
   
 //  }

  int i;
  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, currentSize,nextSize, numSmaller);
  uint numWithDigit, total = 0;
  printf("BEFORE\n");
  partition<T,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock>>>(d_vec, alt_vec, currentSize,
                                                                            digitCounts, totalThreads, previousDigit,numSmaller);
  printf("AFTER\n");

  currentSize = nextSize;
  // for(i =0; i <16; i++){
  //   numWithDigit = countEm<uint,BIT_SHIFT>(d_vec,currentSize, i);
  //   printf("NUM WITH DIGIT %d: %d\n",i, numWithDigit);
  //   total += numWithDigit;
  //   printf("TOTAL: %d\n", total);
  // }
  //  numWithDigit = countEm<uint,BIT_SHIFT>(d_vec,currentSize, currentDigit);
  // printf("NUM WITH DIGIT %d\n", numWithDigit);
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  return currentDigit;
}

uint runDigitPasses(uint *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){
  uint answerSoFar = 0, numSmaller = 0;
  uint *digitCounts, *altDigitCounts, *altVector;
  uint nextSize = size;
  uint countSize =  (16 * blocks * threadsPerBlock) + 1;
  cudaMalloc(&altVector, size * sizeof(uint));
  cudaMalloc(&digitCounts, countSize * sizeof(uint));
  cudaMalloc(&altDigitCounts, countSize * sizeof(uint));

  uint previousDigit;
  previousDigit=  digitPass<uint,4,28>(d_vec,altVector, size,nextSize, k, answerSoFar, digitCounts, altDigitCounts,blocks, threadsPerBlock,countSize, numSmaller,0);
  previousDigit=  digitPass<uint,4,24>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock, countSize,numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,20>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, countSize,numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,16>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock,countSize, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,12>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, countSize,numSmaller,previousDigit);

  previousDigit=  digitPass<uint,4,8>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock,countSize, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,4>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, countSize,numSmaller,previousDigit);

  previousDigit=  digitPass<uint,4,0>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock,countSize, numSmaller,previousDigit);


  cudaFree(altVector);
  cudaFree(altDigitCounts);
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
