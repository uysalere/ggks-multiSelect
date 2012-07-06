#include <thrust/scan.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>
  #include <thrust/reduce.h>
#include "radixsort_key_conversion.h"
namespace DestructiveRadixSelect{

template<typename T>
struct problemInfo_t{
  T *d_vec;
  uint blocks;
  uint threadsPerBlock;
  uint totalThreads;
  uint currentSize;
  uint nextSize;
  uint k;
  uint countSize;
  uint numSmaller;
  uint previousDigit;
  };

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
__global__ void getCounts(T *d_vec,const uint size, uint *digitCounts,const uint offset){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;

  extern __shared__ ushort sharedCounts[];

  for(i = 0; i < 16;i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  //We only look at the current digit, becasue it must be the case that 
  //all elements in d_vec share have the same digit for all digit places before the current one
  for(i = idx; i < size; i += offset){
    sharedCounts[(d_vec[i] >> BIT_SHIFT) * blockDim.x + threadIdx.x]++;
  }

  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
  __global__ void getCounts2(T *d_vec,const uint size, uint *digitCounts,const uint offset,
                             uint answerSoFar){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;

  extern __shared__ ushort sharedCounts[];
  T value;
  for(i = 0; i < 16;i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  //We only look at the current digit, becasue it must be the case that 
  //all elements in d_vec share have the same digit for all digit places before the current one
  for(i = idx; i < size; i += offset){
    value = d_vec[i];
    if((value >> (BIT_SHIFT + 4)) == (answerSoFar >> (BIT_SHIFT + 4))){
      sharedCounts[((d_vec[i] >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;
    }
  }
  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
  __global__ void getCountsWithShrink(T *d_vec,T * alt_vec,const uint size, uint *digitCounts,const uint offset,uint numSmaller,uint previousDigit, uint answerSoFar){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;
  T value;
  int startIndex = digitCounts[ previousDigit * offset + idx] - numSmaller;
  extern __shared__ ushort sharedCounts[];

  for(i = 0; i < 16;i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  for(i = idx; i < size; i += offset){
      value = d_vec[i];
      if( (value >> (BIT_SHIFT + 4))  == (answerSoFar >> (BIT_SHIFT + 4))){

       alt_vec[startIndex++] = value;
        sharedCounts[((value >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;

      }
  }

  syncthreads();

  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}



  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT, uint UPDATE>
  uint determineDigit(uint *digitCounts, problemInfo_t<T> &info){
  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  uint *ns;
  ns = (uint *) malloc(sizeof(uint));
  uint i=0;
  if(UPDATE){
    info.currentSize = info.nextSize;
  }
  uint threshold = info.nextSize - info.k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  thrust::exclusive_scan(ptr, ptr + (info.totalThreads * 16)+1, ptr);
  

  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * info.totalThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }
 

 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    info.k -= info.nextSize - numLessThanOrEqualTo[0];
    info.nextSize = numLessThanOrEqualTo[0];
    info.numSmaller = 0;
    return 0;

  }

  for(i = 1; i < 16; i++){
    if(numLessThanOrEqualTo[i] >= threshold){
      info.k -= info.nextSize - numLessThanOrEqualTo[i];
      info.nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      cudaMemcpy(ns, digitCounts + (i * info.totalThreads) , sizeof(uint), cudaMemcpyDeviceToHost);
      info.numSmaller = ns[0];
      free(ns);
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


  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT,typename PreprocessFunctor >
  void  digitPass(uint *d_vec, uint &answerSoFar, uint *digitCounts,problemInfo_t<T> &info){

  T currentDigit;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * info.threadsPerBlock * sizeof(ushort);

  if(BIT_SHIFT == 28){
    getCounts<T,RADIX_SIZE,BIT_SHIFT><<<info.blocks, info.threadsPerBlock, neededSharedMemory>>>(d_vec, info.currentSize,digitCounts,info.totalThreads);
  }
  else{
    getCounts2<T,RADIX_SIZE,BIT_SHIFT><<<info.blocks, info.threadsPerBlock, neededSharedMemory>>>(d_vec, info.currentSize,digitCounts,info.totalThreads,answerSoFar);
  }

  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT,0>(digitCounts, info);
 
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  info.previousDigit = currentDigit;
} 

  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT,typename PreprocessFunctor >
  void  digitPassWithShrink(uint *d_vec, uint *alt_vec, uint &answerSoFar, uint *digitCounts, problemInfo_t<T> &info){
  T currentDigit;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * info.threadsPerBlock * sizeof(ushort);


  getCountsWithShrink<T,RADIX_SIZE,BIT_SHIFT><<<info.blocks, info.threadsPerBlock, neededSharedMemory>>>(d_vec,alt_vec,
                                                                                                         info.currentSize,
                                                                                                         digitCounts,
                                                                                                         info.totalThreads,
                                                                                                         info.numSmaller,
                                                                                                         info.previousDigit,
                                                                                                         answerSoFar);

  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT,1>(digitCounts, info);
  

  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  info.previousDigit = currentDigit;
} 

template<typename T>
void setupInfo(problemInfo_t<T> &info, uint blocks, uint threadsPerBlock, uint size, uint k, T *d_vec){
    info.blocks = blocks;
    info.threadsPerBlock = threadsPerBlock;
    info.totalThreads = blocks * threadsPerBlock;
    info.currentSize = size;
    info.nextSize = size; 
    info.k = k;
    info.countSize = (16 * blocks * threadsPerBlock) + 1;
    info.numSmaller = 0;
    info.previousDigit = 0;
    info.d_vec = d_vec;
  }

template<typename T>
uint runDigitPasses(T *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){
  uint answerSoFar = 0;
  uint *digitCounts, *altVector;
  problemInfo_t<T> info;
  setupInfo(info,blocks, threadsPerBlock, size, k,d_vec);
  cudaMalloc(&digitCounts, info.countSize * sizeof(uint));
  
  typedef typename KeyConversion<T>::UnsignedBits ConvertedType;
  problemInfo_t<ConvertedType> convertedInfo;
  float time;
  cudaEvent_t start,stop;
 cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  memcpy(&convertedInfo, &info, sizeof(problemInfo_t<T>));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);
  printf("TIME:%f\n", time);
  digitPass<uint,4,28,PreprocessKeyFunctor<T> >(d_vec, answerSoFar, digitCounts,convertedInfo);
  digitPass<uint,4,24,PreprocessKeyFunctor<T> >(d_vec, answerSoFar, digitCounts,convertedInfo);
  cudaMalloc(&altVector, info.nextSize * sizeof(uint));
  digitPassWithShrink<uint,4,20, PreprocessKeyFunctor<T> >(d_vec,altVector, answerSoFar, digitCounts, convertedInfo);  
  digitPass<uint,4,16,NopFunctor<ConvertedType> >(altVector, answerSoFar, digitCounts,convertedInfo);
  digitPassWithShrink<uint,4,12,NopFunctor<ConvertedType> >(altVector,convertedInfo.d_vec,answerSoFar, digitCounts,convertedInfo);
  digitPass<uint,4,8,NopFunctor<ConvertedType> >(d_vec, answerSoFar, digitCounts,convertedInfo);
  digitPassWithShrink<uint,4,4,NopFunctor<ConvertedType>  >(d_vec,altVector, answerSoFar, digitCounts,convertedInfo);
  digitPass<uint,4,0,NopFunctor<ConvertedType> >(altVector, answerSoFar, digitCounts,convertedInfo);
  
  cudaFree(altVector);
  cudaFree(digitCounts);
  return answerSoFar;
}

template<typename T>
T radixSelect(T *d_vec, uint size, uint k){
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0) ;
  uint blocks = dp.multiProcessorCount ;//16;
  uint threadsPerBlock = dp.maxThreadsPerBlock;///2;
  return runDigitPasses<T>(d_vec, size, k, blocks, threadsPerBlock);
}

}
