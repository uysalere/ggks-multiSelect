#include <thrust/scan.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>
  #include <thrust/for_each.h>
#include "radixsort_key_conversion.h"
namespace DestructiveRadixSelect{
#define THREADS_PER_BLOCK 1024
#define BLOCKS 14
#define OFFSET 14336
#define OFFSET 14336
// BIT 28   : 27.264769
// BIT 24   : 32.631454
// BIT 20   : 34.620609

template<typename T>
struct problemInfo_t{
  T *d_vec;
  uint canShrink;
  uint blocks;
  uint threadsPerBlock;
  uint totalThreads;
  uint currentSize;
  uint nextSize;
  uint k;
  uint countSize;
  uint numSmaller;
  uint previousDigit;
  uint beenPreprocessed;
  };

//  struct subtract_x {
//  subtract_x(uint x) : x(x) {}
//     __device__ void operator()(uint y) { y -= x;}

//    uint x;
// };

struct subtract_x {
  subtract_x(uint x) : x(x) {}
    __device__ void operator()(uint &y) {
      y  -=x;
    }
  uint x;
};
 //  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, typename PreprocessFunctor>
 // getCounts(T *d_vec,const uint size, uint *digitCounts,const uint offset,PreprocessFunctor preprocess = PreprocessFunctor() ) {
 //      int idx = blockDim.x * blockIdx.x + threadIdx.x;
 //      int i;
 //      extern __shared__ uint sharedCounts[];
 //      T value;
 //      for(i =0;i <(1<<RADIX_SIZE); i++){
 //        value = d_vec[i];
 //        preprocess(value);
 //        shared
 //  extern __shared__ ushort sharedCounts[];x
  //31.39
  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT,typename PreprocessFunctor>
  __global__ void  getCounts(const T *d_vec,const uint size, uint *digitCounts,PreprocessFunctor preprocess = PreprocessFunctor()){ 
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;

   __shared__ ushort sharedCounts[16384];

  T value;
#pragma unroll
  for(i = 0; i < (1 <<RADIX_SIZE);i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }
 
  for(i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += OFFSET ){
    value = d_vec[i];
    preprocess(value);
    sharedCounts[ (value>> BIT_SHIFT) * blockDim.x + threadIdx.x]++;
  }

#pragma unroll
  for(i = 0; i < (1 <<(RADIX_SIZE)) ;i++){
    digitCounts[i *OFFSET + (blockDim.x * blockIdx.x + threadIdx.x)] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  }

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT,uint MASK, typename PreprocessFunctor>
  __global__ void getCounts2( T *d_vec, uint size, uint *digitCounts,
                             uint answerSoFar,PreprocessFunctor preprocess = PreprocessFunctor()){ 
    //int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;

   __shared__ ushort sharedCounts[16384];
#pragma unroll
  for(i = 0; i < (1 <<RADIX_SIZE);i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  //We only look at the current digit, becasue it must be the case that 
  //all elements in d_vec share have the same digit for all digit places before the current one
  for(i =  blockDim.x * blockIdx.x + threadIdx.x; i < size; i += OFFSET){
   T value = d_vec[i];
     preprocess(value);
    if((value >> BIT_SHIFT_PREV) == answerSoFar ){
      sharedCounts[((value >> BIT_SHIFT) & MASK ) * blockDim.x + threadIdx.x]++;
    }
  }
#pragma unroll
  for(i = 0; i <(1 <<RADIX_SIZE) ;i++){
    digitCounts[i * OFFSET + (blockDim.x * blockIdx.x + threadIdx.x)] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}


  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT,uint BIT_SHIFT_PREV,uint MASK, typename PreprocessFunctor>
  __global__ void getCountsWithShrink(const T *d_vec,T * alt_vec,const uint size, uint *digitCounts,
                                       uint answerSoFar,PreprocessFunctor preprocess = PreprocessFunctor()){ 
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;
  T value;
  int startIndex = digitCounts[(answerSoFar & MASK) * OFFSET + (blockDim.x * blockIdx.x + threadIdx.x)];
   __shared__ ushort sharedCounts[16384];

#pragma unroll
  for(i = 0; i < (1 <<RADIX_SIZE);i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  for(i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += OFFSET){
    value = d_vec[i];
    preprocess(value);

    if( (value >>  BIT_SHIFT_PREV)  == answerSoFar){

       alt_vec[startIndex++] = value;
        sharedCounts[((value >> BIT_SHIFT) & MASK) * blockDim.x + threadIdx.x]++;

      }
  }

#pragma unroll
  for(i = 0; i <(1 <<RADIX_SIZE) ;i++){
    digitCounts[i * OFFSET + (blockDim.x * blockIdx.x + threadIdx.x)] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  
  }


 template<typename T,uint RADIX_SIZE, uint BIT_SHIFT, uint UPDATE>
  uint determineDigit(uint *digitCounts, problemInfo_t<T> &info){
   subtract_x subtractFunctor(0);

  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc((1 <<RADIX_SIZE) * sizeof(uint));
  uint *ns;
  ns = (uint *) malloc(sizeof(uint));
  uint i=0;
  if(UPDATE){
    info.currentSize = info.nextSize;
  }
  uint threshold = info.nextSize - info.k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  thrust::exclusive_scan(ptr, ptr + (info.totalThreads * (1 <<RADIX_SIZE)) +1, ptr);
  

  for(i =0; i <(1 <<RADIX_SIZE); i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * info.totalThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }
 
 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    info.k -= info.nextSize - numLessThanOrEqualTo[0];
    info.nextSize = numLessThanOrEqualTo[0];
    info.numSmaller = 0;
    return 0;

  }

  for(i = 1; i < (1 <<RADIX_SIZE); i++){
    if(numLessThanOrEqualTo[i] >= threshold){
      info.k -= info.nextSize - numLessThanOrEqualTo[i];
      info.nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      cudaMemcpy(ns, digitCounts + (i * info.totalThreads) , sizeof(uint), cudaMemcpyDeviceToHost);
      info.numSmaller = ns[0];
      subtractFunctor.x = ns[0];
      thrust::for_each(ptr, ptr + (info.totalThreads * (1 <<RADIX_SIZE)) +1,subtractFunctor);
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


  template<typename T,uint RADIX_SIZE, uint BIT_SHIFT, uint BIT_SHIFT_PREV,typename PreprocessFunctor >
  void  digitPass(uint &answerSoFar, uint *digitCounts,T *alt_vec,T *alt_vec2,problemInfo_t<T> &info){

 
    const  uint MASK = 0xFFFFFFFF >> (32 - RADIX_SIZE);
  float time;
  cudaEvent_t start, stop;
 cudaEventCreate(&start);
  cudaEventCreate(&stop);
  info.countSize = ((1 << RADIX_SIZE) * info. blocks * info.threadsPerBlock) + 1;
  T currentDigit;
  // uint neededSharedMemory = ((1 << RADIX_SIZE) )  * info.threadsPerBlock * sizeof(ushort);
  cudaEventRecord(start,0);
  if(BIT_SHIFT == 28){
    getCounts<T,RADIX_SIZE,BIT_SHIFT,PreprocessFunctor><<<info.blocks, info.threadsPerBlock>>>(info.d_vec, info.currentSize,digitCounts);
    currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT,0>(digitCounts, info);
    info.canShrink = 1;
  }
  else if(info.nextSize < (info.currentSize / 32) && info.canShrink){

    if(alt_vec == NULL){
      cudaMalloc(&alt_vec, info.nextSize * sizeof(uint));
     }
    if(alt_vec2 == NULL && alt_vec != NULL){
      cudaMalloc(&alt_vec2, info.nextSize * sizeof(uint));
      alt_vec = alt_vec2;
     }
    if(!info.beenPreprocessed){
      getCountsWithShrink<T,RADIX_SIZE,BIT_SHIFT, BIT_SHIFT_PREV,MASK,PreprocessFunctor><<<info.blocks, info.threadsPerBlock>>>(info.d_vec,alt_vec,
                                                                                                         info.currentSize,
                                                                                                         digitCounts,
                                                                                                         answerSoFar >> BIT_SHIFT_PREV);
    }
    else{

      getCountsWithShrink<T,RADIX_SIZE,BIT_SHIFT, BIT_SHIFT_PREV,MASK,NopFunctor<uint> ><<<info.blocks, info.threadsPerBlock>>>(info.d_vec,alt_vec,
                                                                                                         info.currentSize,
                                                                                                         digitCounts,
                                                                                                         answerSoFar>> BIT_SHIFT_PREV);
    }
    // printf("NEXT:\n");
    // PrintFunctions::printCudaArrayBinary(alt_vec,10);
    info.beenPreprocessed = 1;
    info.canShrink = 0;
    T *temp;
    temp = info.d_vec;
    info.d_vec = alt_vec;
    alt_vec = temp;
    currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT,1>(digitCounts, info);
  }
  else{
    if(!info.beenPreprocessed){
      getCounts2<T,RADIX_SIZE,BIT_SHIFT, BIT_SHIFT_PREV,MASK,PreprocessFunctor><<<info.blocks, info.threadsPerBlock>>>(info.d_vec, info.currentSize,digitCounts,
                                                                                                                       answerSoFar>> BIT_SHIFT_PREV);
    }
    else{
      getCounts2<T,RADIX_SIZE,BIT_SHIFT, BIT_SHIFT_PREV,MASK,NopFunctor<uint> ><<<info.blocks, info.threadsPerBlock>>>(info.d_vec, info.currentSize,
                                                                                                                       digitCounts,answerSoFar >> BIT_SHIFT_PREV);
    }
      currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT,0>(digitCounts, info);
      info.canShrink = 1;
    
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);
  printf("BIT %d   : %f\n", BIT_SHIFT,time);

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
    info.countSize = (64 * blocks * threadsPerBlock) + 1;
    info.numSmaller = 0;
    info.previousDigit = 0;
    info.d_vec = d_vec;
    info.beenPreprocessed  =0;
    info.canShrink = 0;
  }

template<typename T>
T runDigitPasses(T *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){
  uint answerSoFar = 0;
  T answer;
  uint *digitCounts;
  problemInfo_t<uint> info;
  uint *alt_vec = NULL;
  uint *alt_vec2 = NULL;

  uint *convertedD_vec = (uint *)d_vec;
  //   FILE *fp;
  //   fp = fopen("cleanOutput.txt", "w");
  // fclose(fp);
  setupInfo<uint>(info,blocks, threadsPerBlock, size, k,convertedD_vec);
  cudaMalloc(&digitCounts, info.countSize * sizeof(uint));
  
  typedef typename KeyConversion<T>::UnsignedBits ConvertedType;
  PostprocessKeyFunctor<T> postprocess = PostprocessKeyFunctor<T>();
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

   //FOR UINTS;
  digitPass<uint,4,28,28,PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,24,28,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,20,24,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,16,20, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,alt_vec2, info);  
  digitPass<uint,4,12,16, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,8,12, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,4,8, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,alt_vec2,info);
  digitPass<uint,4,0,4, PreprocessKeyFunctor<T>  >(answerSoFar, digitCounts,alt_vec,alt_vec2,info);

   // digitPass<uint,6,26,28,PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,6,20,26,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,6,14,20,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,6,8,14, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec, info);  
   // digitPass<uint,6,2,8, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,2,0,2, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   

   // digitPass<uint,3,29,28,PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,26,29,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,23,26,PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,20,23, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec, info);  
   // digitPass<uint,3,17,20, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,14,17, PreprocessKeyFunctor<T> >(answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,11,14, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,8,11, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,5,8, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,3,2,5, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,2,0,2, PreprocessKeyFunctor<T> >( answerSoFar, digitCounts,alt_vec,info);
   // digitPass<uint,4,0,4, PreprocessKeyFunctor<T>  >(answerSoFar, digitCounts,alt_vec,info);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);
  printf("   : %f\n",time);
 
  cudaFree(alt_vec);
  cudaFree(alt_vec2);

  cudaFree(digitCounts);
  postprocess(answerSoFar);
  memcpy(&answer, &answerSoFar, sizeof(T));
  return answer;
}

template<typename T>
T radixSelect(T *d_vec, uint size, uint k){
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0) ;
  uint blocks = dp.multiProcessorCount;
  uint threadsPerBlock = dp.maxThreadsPerBlock;
  return runDigitPasses<T>(d_vec, size, k, blocks, threadsPerBlock);
}

}
