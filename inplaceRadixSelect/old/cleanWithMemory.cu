#include <thrust/scan.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>

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
__global__ void getCounts(T *d_vec,const uint size, uint *digitCounts,const uint offset){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;

  extern __shared__ ushort sharedCounts[];

  for(i = 0; i < 16;i++){
    sharedCounts[i * blockDim.x + threadIdx.x] = 0;
  }

  for(i = idx; i < size; i += offset){
    sharedCounts[((d_vec[i] >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;
  }



  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}

template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
__global__ void getCounts2(T *d_vec,T * alt_vec,const uint size, uint *digitCounts,const uint offset,uint numSmaller, uint previousDigit, uint *replaced){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i,j;
  T value;
  int startIndex = digitCounts[ previousDigit * offset + idx] - numSmaller;
  extern __shared__ ushort sharedCounts[];
  //  alt_vec[startIndex] =15;

  for(j = 0; j < 16;j++){
    sharedCounts[j * blockDim.x + threadIdx.x] = 0;
  }

  for(i = idx; i < size; i += offset){
      value = d_vec[i];
      //replaced[i]++;// = 1;
      if(((value >> (BIT_SHIFT + 4)) & 0xF) == previousDigit){

       alt_vec[startIndex++] = value;
        sharedCounts[((value >> BIT_SHIFT) & 0xF) * blockDim.x + threadIdx.x]++;

      }
      if(((value >> (BIT_SHIFT + 4)) & 0xF) == previousDigit){
                  replaced[i] = idx;
      }
  }

  syncthreads();

  for(i = 0; i <16 ;i++){
    digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
  }
  

}



template<typename T, uint RADIX_SIZE, uint BIT_SHIFT>
__global__ void partition(uint * d_vec, uint *alt_vec, const uint size,const uint *digitCounts,const uint offset,const uint digit,const uint numSmaller){
       const int idx = blockDim.x * blockIdx.x + threadIdx.x;
       int startIndex = digitCounts[ digit * offset + idx] - numSmaller;
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
uint determineDigit(uint *digitCounts, uint &k, uint countSize,uint numThreads, uint &currentSize,uint &nextSize,uint &numSmaller,FILE *fp){
  uint *numLessThanOrEqualTo;
  numLessThanOrEqualTo = (uint *) malloc(16 * sizeof(uint));
  uint *ns;
  ns = (uint *) malloc(sizeof(uint));
  uint i=0;
  uint threshold = nextSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  currentSize = nextSize;
  thrust::exclusive_scan(ptr, ptr + (numThreads * 16)+1, ptr);
 

  for(i =0; i < 16; i++){
    cudaMemcpy(numLessThanOrEqualTo + i, digitCounts + ((i+1) * numThreads), sizeof(uint), cudaMemcpyDeviceToHost);
  }
 //identify the kth largest digit
  if(numLessThanOrEqualTo[0] >= threshold){
    k -= currentSize - numLessThanOrEqualTo[0];
    nextSize = numLessThanOrEqualTo[0];
    numSmaller = 0;
    return 0;

  }

  for(i = 1; i < 16; i++){
    //  printf("%d   %d\n", i, numLessThanOrEqualTo[i]);
    if(numLessThanOrEqualTo[i] >= threshold){
      // printf("Offsets 2\n");
      // PrintFunctions::printCudaArray(digitCounts + (i * numThreads), 10);
      k -= currentSize - numLessThanOrEqualTo[i];
      nextSize = numLessThanOrEqualTo[i] - numLessThanOrEqualTo[i -1];
      cudaMemcpy(ns, digitCounts + (i * numThreads) , sizeof(uint), cudaMemcpyDeviceToHost);
      numSmaller = ns[0];
      free(ns);
      return i;
    }
  }
  fprintf(fp,"OPPS\n");
  return 5367;
}


template<typename T,uint BIT_SHIFT>
void updateAnswerSoFar(T &answerSoFar,T digitValue){
  T digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}


template<typename T,uint RADIX_SIZE, uint BIT_SHIFT>
T  digitPass2(uint *d_vec, uint *alt_vec, uint &currentSize,uint &nextSize, uint &k, uint &answerSoFar, uint *digitCounts, uint blocks,uint threadsPerBlock,uint countSize,  uint &numSmaller,uint previousDigit){
  FILE *fp;
  fp = fopen("cleanOutput.txt", "a");
  uint *replacedVec;
  cudaMalloc(&replacedVec, currentSize * sizeof(uint));
  printf("num trheads: %d\n",threadsPerBlock );
  uint oldSize = currentSize;
  fprintf(fp,"BIT: %d\n", BIT_SHIFT);
  fprintf(fp,"CUR SIZE: %d\nnextSize: %d\nk: %d\nPrevious digit: %d\n", currentSize, nextSize, k, previousDigit);
  fprintf(fp,"num smaller: %d\n", numSmaller);
cudaMemset(replacedVec,0, currentSize * sizeof(uint));
  //uint numSmaller;
 fprintf(fp,"Incoming\n");
 PrintFunctions::printCudaArrayBinaryFile(d_vec, currentSize,fp);
  T currentDigit;
  uint totalThreads = blocks* threadsPerBlock;
  uint neededSharedMemory = ((1 << RADIX_SIZE) ) * threadsPerBlock * sizeof(ushort);
  //  cudaMemset(digitCounts,0, countSize * sizeof(uint));
  if(BIT_SHIFT == 28){
      getCounts<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec, currentSize,digitCounts,totalThreads);
  }
  else{
    printf("Offsets before \n");
    PrintFunctions::printCudaArrayFile(digitCounts + (totalThreads * previousDigit),threadsPerBlock,fp);
    getCounts2<T,RADIX_SIZE,BIT_SHIFT><<<blocks, threadsPerBlock, neededSharedMemory>>>(d_vec,alt_vec, currentSize,digitCounts,totalThreads,numSmaller,previousDigit,replacedVec);
    fprintf(fp,"NUMBER TRANSFERED: %d \n",countEm<uint, BIT_SHIFT + 4>(d_vec, currentSize, previousDigit));
    thrust::device_ptr<uint>ptr(replacedVec);
    fprintf(fp,"TOTAL COUNTED: %d\n", thrust::reduce(ptr, ptr + currentSize));
    fprintf(fp,"NEXT\n");
    PrintFunctions::printCudaArrayBinaryFile(replacedVec, currentSize,fp);
    fprintf(fp,"NEXT\n");
    PrintFunctions::printCudaArrayBinaryFile(alt_vec, nextSize,fp);
  }

  fprintf(fp,"After counts\n");
  currentDigit = determineDigit<T,RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, totalThreads, currentSize,nextSize,numSmaller,fp);
  fprintf(fp,"OFFICIAL COUNTS: %d \n",countEm<uint, BIT_SHIFT+4>(d_vec, oldSize, currentDigit));
  thrust::device_ptr<uint>ptr(replacedVec);
  fprintf(fp,"After determine %d\n",currentDigit);
  updateAnswerSoFar<T,BIT_SHIFT>(answerSoFar, currentDigit);
  cudaFree(replacedVec);
fclose(fp);
  return currentDigit;
} 
uint runDigitPasses(uint *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){
  uint answerSoFar = 0, nextSize = size, previousDigit, numSmaller = 0;
  uint *digitCounts, *altVector;
  uint countSize =  (16 * blocks * threadsPerBlock) + 1;
  cudaMalloc(&altVector, size * sizeof(uint));
  cudaMalloc(&digitCounts, countSize * sizeof(uint));
 FILE *fp;
  fp = fopen("cleanOutput.txt", "w");
  fclose(fp);


  printf("INITIAL\n");
  PrintFunctions::printCudaArrayBinary(d_vec, 10);
  previousDigit =   digitPass2<uint,4,28>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, 0);
  previousDigit =   digitPass2<uint,4,24>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);  
  previousDigit =  digitPass2<uint,4,20>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass2<uint,4,16>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass2<uint,4,12>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass2<uint,4,8>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass2<uint,4,4>(altVector,d_vec, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);
  previousDigit =    digitPass2<uint,4,0>(d_vec,altVector, size,nextSize,k, answerSoFar, digitCounts,blocks, threadsPerBlock,countSize,numSmaller, previousDigit);


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
