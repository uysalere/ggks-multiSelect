#include <thrust/scan.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>

namespace CountingRadixSelect{


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
    sharedCounts[((d_vec[i] >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;
  }

  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}
template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
__global__ void getCounts2(T *d_vec,const uint size, uint *digitCounts,const uint offset, uint answerSoFar){ 
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



template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
uint determineDigit(uint *digitCounts, uint &k, uint countSize,uint numThreads, uint currentSize,uint &nextSize,uint &numSmaller){
  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  uint *ns;
  ns = (uint *) malloc(sizeof(uint));
  uint i=0;
  uint threshold = nextSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  thrust::exclusive_scan(ptr, ptr + (numThreads * 16)+1, ptr);
  //   fprintf(fp,"THRESHOLD: %d\n", threshold);


  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }
  // fprintf(fp, "0   %d\n", numLessThanOrEqualTo[0]);

 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    k -= nextSize - numLessThanOrEqualTo[0];
    nextSize = numLessThanOrEqualTo[0];
    numSmaller = 0;
    return 0;

  }

  for(i = 1; i < 16; i++){
    // fprintf(fp, "%d   %d\n", i, numLessThanOrEqualTo[i]);

    if(numLessThanOrEqualTo[i] >= threshold){
      k -= nextSize - numLessThanOrEqualTo[i];
      nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      cudaMemcpy(ns, digitCounts + (i * numThreads) , sizeof(uint), cudaMemcpyDeviceToHost);
      numSmaller = ns[0];
      free(ns);
      return i;
    }
  }
  printf("OPPS\n");
  return 5367;
}

template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
uint determineDigitWithShrink(uint *digitCounts, uint &k, uint countSize,uint numThreads, uint &currentSize,uint &nextSize,uint &numSmaller){
  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  uint *ns;
  ns = (uint *) malloc(sizeof(uint));
  uint i=0;
  uint threshold = nextSize - k + 1;
  // fprintf(fp,"THRESHOLD: %d\n", threshold);
  thrust::device_ptr<uint>ptr(digitCounts);
  currentSize = nextSize;
  thrust::exclusive_scan(ptr, ptr + (numThreads * 16)+1, ptr);
 

  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }   
  // fprintf(fp, "0   %d\n", numLessThanOrEqualTo[0]);

 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    k -= currentSize - numLessThanOrEqualTo[0];
    nextSize = numLessThanOrEqualTo[0];
    numSmaller = 0;
    return 0;

  }

  for(i = 1; i < 16; i++){
    // fprintf(fp, "%d   %d\n", i, numLessThanOrEqualTo[i]);
    if(numLessThanOrEqualTo[i] >= threshold){
      k -= currentSize - numLessThanOrEqualTo[i];
      nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      cudaMemcpy(ns, digitCounts + (i * numThreads) , sizeof(uint), cudaMemcpyDeviceToHost);
      numSmaller = ns[0];
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


template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T  digitPass(uint *d_vec, uint currentSize,uint &nextSize, uint &k, uint &answerSoFar, uint *digitCounts, uint blocks,uint threadsPerBlock,uint countSize,  uint &numSmaller,uint previousDigit){
   
  // FILE *fp;
  // fp = fopen("cleanOutput.txt", "a");
  // fprintf(fp,"BIT: %d\n", BIT_SHIFT);
  // fprintf(fp,"CUR SIZE: %d\nnextSize: %d\nk: %d\nPrevious digit: %d\n", currentSize, nextSize, k,previousDigit);
  // fprintf(fp,"num smaller: %d\n", numSmaller);
  // fprintf(fp,"Incoming\n");
  // PrintFunctions::printCudaArrayBinaryFile(d_vec, currentSize,fp); 
  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * threadsPerBlock * sizeof(ushort);
  if(BIT_SHIFT == 28){
    getCounts<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, currentSize,digitCounts,totalThreads);
  }
  else{
    getCounts2<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, currentSize,digitCounts,totalThreads,answerSoFar);
  }
  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, currentSize,nextSize,numSmaller);

  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  // fclose(fp);

  return currentDigit;
} 

template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T  digitPassWithShrink(uint *d_vec, uint *alt_vec, uint &currentSize,uint &nextSize, uint &k, uint &answerSoFar, uint *digitCounts, uint blocks,uint threadsPerBlock,uint countSize,  uint &numSmaller,uint previousDigit){
  // FILE *fp;
  // fp = fopen("cleanOutput.txt", "a");
  // fprintf(fp,"BIT: %d\n", BIT_SHIFT);
  // fprintf(fp,"CUR SIZE: %d\nnextSize: %d\nk: %d\nPrevious digit: %d\n", currentSize, nextSize, k, previousDigit);
  // fprintf(fp,"num smaller: %d\n", numSmaller);
  // fprintf(fp,"Incoming\n");
  // PrintFunctions::printCudaArrayBinaryFile(d_vec, currentSize,fp);
  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * threadsPerBlock * sizeof(ushort);
  
  getCountsWithShrink<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec,alt_vec, currentSize,digitCounts,totalThreads,numSmaller,previousDigit,answerSoFar);
  // fprintf(fp,"NEXT\n");
  // PrintFunctions::printCudaArrayBinaryFile(alt_vec, nextSize,fp);


  currentDigit = determineDigitWithShrink<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, currentSize,nextSize,numSmaller);
  
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  // fclose(fp);

  return currentDigit;
} 


uint runDigitPasses(uint *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){
  uint answerSoFar = 0, nextSize = size, previousDigit, numSmaller = 0;
  uint *digitCounts, *altVector;
  uint countSize =  (16 * blocks * threadsPerBlock) + 1;
  cudaMalloc(&altVector, size * sizeof(uint));
  cudaMalloc(&digitCounts, countSize * sizeof(uint));
  //T  digitPass(uint *d_vec, uint currentSize, uint &k, uint &answerSoFar, uint *digitCounts, uint blocks,uint threadsPerBlock,uint countSize,  uint &numSmaller){

  // FILE *fp;
//   fp = fopen("cleanOutput.txt", "w");
// fclose(fp);

 previousDigit =   digitPass<uint,4,28>(d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller,0);
  previousDigit =   digitPassWithShrink<uint,4,24>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);  
  previousDigit =  digitPass<uint,4,20>(altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller,previousDigit);
  previousDigit =    digitPassWithShrink<uint,4,16>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass<uint,4,12>(d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller,previousDigit);
  previousDigit =    digitPassWithShrink<uint,4,8>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass<uint,4,4>(altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller,previousDigit);
  previousDigit =    digitPass<uint,4,0>(altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller,previousDigit);


  cudaFree(altVector);
  cudaFree(digitCounts);
  return answerSoFar;
}
  // I need to alternate counting, with the counting/redistribute step
template<typename T>
T countingRadixSelect(T *d_vec, uint size, uint k){
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  uint blocks = dp.multiProcessorCount;
  uint threadsPerBlock = dp.maxThreadsPerBlock;
  return runDigitPasses(d_vec, size, k, blocks, threadsPerBlock);
}

}
