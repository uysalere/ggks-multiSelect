#include <thrust/scan.h>

namespace CountingRadixSelect{

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
    uint startCount = altDigitCounts[ previousDigit * offset + idx] - altDigitCounts[ previousDigit * offset];//numSmaller;
    T value, value2;
    extern __shared__ ushort sharedCounts[];
    reduced_vec[startCount] = 15;

    for(j = 0; j < 16;j++){
      sharedCounts[j * blockDim.x + threadIdx.x] = 0;
    }
    for(i = idx; i < size; i += offset){
      value = d_vec[i];
      value2 = value & MASK0;
      if(((value >> BIT_SHIFT) & 0xF) ==previousDigit){
      if(value2 >= answerSoFar && (value2 <= max)){
         sharedCounts[((value2 >> BIT_SHIFT) & mask1) * blockDim.x + threadIdx.x]++;
         reduced_vec[startCount] = 0;
         reduced_vec[startCount] = value;
         startCount += 1;
      }
    }
    syncthreads();
    for(i = 0; i < 16 ;i++){
      //  if(sharedCounts[blockDim.x * i + threadIdx.x]){
      digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
        // }
    }
  }
template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, T MASK0>
__global__ void partition(uint * d_vec, uint *alt_vec, uint size, uint *digitCounts, uint offset, uint digit, uint * vectorOfOffsets, uint numSmaller){
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       uint startCount2 = digitCounts[ digit * offset + idx];// - digitCounts[ digit * offset];//numSmaller;

       uint startIndex = digitCounts[ digit * offset + idx] - numSmaller;//digitCounts[ digit * offset];//numSmaller;
       vectorOfOffsets[idx] = startIndex;
       int i;
       T value;
       for(i = idx; i < size; i += offset){
         value = d_vec[i];
         if(((value >> BIT_SHIFT) & 0xF) == digit){
           alt_vec[startIndex++] = value;
         }
       }
}

template<typename T,uint BIT_SHIFT>
void updateAnswerSoFar(T &answerSoFar,T digitValue){
  T digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}
  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T determineDigit(uint *digitCounts, uint &k, uint countSize,uint numThreads, uint &currentSize,uint &nextSize, uint &numSmaller){
  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  uint i=0;
  uint *ns;
  ns = (uint *)malloc(sizeof(uint));
  currentSize = nextSize;
  uint threshold = nextSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  PrintFunctions::printCudaArray(digitCounts + (15 * numThreads), 10);
  thrust::exclusive_scan(ptr, ptr + (numThreads * 16)+1, ptr);
  PrintFunctions::printCudaArray(digitCounts + (15 * numThreads)+ numThreads - 10, 11);

  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + (i * numThreads + (numThreads - 1))+1, sizeof(uint), cudaMemcpyDeviceToHost);
  }
  
 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    k -= nextSize - numLessThanOrEqualTo[0];
    nextSize = numLessThanOrEqualTo[0];
    numSmaller = 0;
  }
  for(i = 1; i < 16; i++){
      printf("%d   %d\n", i , numLessThanOrEqualTo[i]);
    if(numLessThanOrEqualTo[i] >= threshold){
      k -= nextSize - numLessThanOrEqualTo[i];
      cudaMemcpy(ns, digitCounts + ((i-1) * numThreads + numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
      nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      // numSmaller = numLessThanOrEqualTo[i -1];
      numSmaller= ns[0];
      return i;
    }
  }
  printf("OPPS\n");
  return 5367;
}

template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T  digitPass(T *d_vec,T *nextVector,uint &currentSize,uint &nextSize,uint &k, T &answerSoFar, uint *digitCounts,uint *altDigitCounts, uint blocks, uint threadsPerBlock, uint &numSmaller, uint previousDigit){
  printf("\n\n\nBIT_SHIFT: %d\nCSIZE:%d\nNSIZE:%d\nK:%d\nPREVIOUS:%d\nNUM SMALLER:%d\n",BIT_SHIFT,currentSize,nextSize,k, previousDigit,numSmaller);
  printf("INCOMING\n");
  PrintFunctions::printCudaArrayBinary(d_vec, 10);
  T x = (T ((1 << RADIX_SIZE) - 1)) << BIT_SHIFT;
  printf("ANSWER SO FAR\n");
  PrintFunctions::printBinary(answerSoFar);
 
  const T mask0 = ~( ((T) 1 << BIT_SHIFT) - 1);
  T  mask1 = (1 << (RADIX_SIZE)) - 1;


  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint countSize = 16 * blocks * threadsPerBlock +1; 
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * threadsPerBlock * sizeof(ushort);
  cudaMemset(digitCounts,0, countSize * sizeof(uint));

  if(BIT_SHIFT == 28){
    getCountsFirstPass<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, currentSize,digitCounts,totalThreads);
  }
  else{
    uint *vectorOfOffsets;
    uint *displayVec;
    cudaMalloc(&displayVec, nextSize * sizeof(uint));
    cudaMalloc(&vectorOfOffsets, totalThreads * sizeof(uint));
    //      __global__ void partition(uint * d_vec, uint *alt_vec, uint size, uint *digitCounts, uint offset, uint digit){
    partition<T,RADIX_SIZE, BIT_SHIFT +4, mask0><<<blocks, threadsPerBlock>>>(d_vec, displayVec, currentSize, altDigitCounts, totalThreads, previousDigit,vectorOfOffsets,numSmaller);
    printf("DISPLAY VECTOR\n");
     PrintFunctions::printCudaArrayBinary(displayVec, 10);
     printf("VECTOR Of Offsets\n");
     PrintFunctions::printCudaArray(vectorOfOffsets, 10);
     printf("VECTOR Of Offsets2\n");
     PrintFunctions::printCudaArray(altDigitCounts +(previousDigit * totalThreads), 10);
    cudaFree(displayVec);
    cudaFree(vectorOfOffsets);

    getCounts<T,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock,neededSharedMemory>>>(d_vec,nextVector, currentSize, answerSoFar, digitCounts,
                                                                                    altDigitCounts, mask1,totalThreads, answerSoFar | x, previousDigit, numSmaller);
     printf("NEXT\n");
     PrintFunctions::printCudaArrayBinary(nextVector, 10);
    //PrintFunctions::printCudaArray(digitCounts, 10);
    printf("BYE\n");
  }

  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, currentSize,nextSize, numSmaller);
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  return currentDigit;
}


uint runDigitPasses(uint *d_vec, uint size, uint k,uint blocks, uint threadsPerBlock){
  uint  numSmaller = 0, answerSoFar = 0;
  uint *digitCounts, *altDigitCounts;
  uint *altVector;
  uint nextSize = size;
  cudaMalloc(&altVector, size * sizeof(uint));
  uint countSize = 16 * blocks * threadsPerBlock;
  cudaMalloc(&digitCounts,( countSize +1) * sizeof(uint));
  cudaMalloc(&altDigitCounts, (1 + countSize) * sizeof(uint));
  uint previousDigit;
  previousDigit=  digitPass<uint,4,28>(d_vec,altVector, size,nextSize, k, answerSoFar, digitCounts, altDigitCounts,blocks, threadsPerBlock, numSmaller,0);
  previousDigit=  digitPass<uint,4,24>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,20>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,16>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,12>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);

  previousDigit=  digitPass<uint,4,8>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);
  previousDigit=  digitPass<uint,4,4>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,altDigitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);

  previousDigit=  digitPass<uint,4,0>(d_vec,altVector, size,nextSize,k, answerSoFar, altDigitCounts,digitCounts, blocks, threadsPerBlock, numSmaller,previousDigit);

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
