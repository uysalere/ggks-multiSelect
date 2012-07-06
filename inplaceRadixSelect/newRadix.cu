
#include <thrust/scan.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>
  #include <thrust/for_each.h>
#include "radixsort_key_conversion.h"
namespace DestructiveRadixSelect{

struct problemInfo_t{
  uint *d_vec;
  uint blocks;
  uint threadsPerBlock;
  uint totalThreads;
  uint curSize;
  uint size;
  uint k;
  uint countSize;
  uint previousDigit;
  };   

template<uint BIT_SHIFT>
__global__ void getCounts(uint *d_vec,const uint size, uint *digitCounts,const uint offset,uint answerSoFar, uint countSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // __shared__ uint counts[32];
  // counts[threadIdx.x] = 0;
  int i;
  uint count[3];// =0;
  count[0] = 0;
  count[1] = 0;
  count[2] = 0;
  for(i =idx;i < size; i += offset){
    count[0] += __popc(__ballot( (d_vec[i] >> BIT_SHIFT) == (answerSoFar >> BIT_SHIFT)));
    count[1] += __popc(__ballot( (d_vec[i] >> BIT_SHIFT) == ((answerSoFar >> BIT_SHIFT) | 0x1 )));
    count[2] += __popc(__ballot( (d_vec[i] >> BIT_SHIFT) ==( (answerSoFar >> BIT_SHIFT) | 0x2 )));
    //    count[0] += __popc(__ballot( (d_vec[i] >> BIT_SHIFT) == (answerSoFar >> BIT_SHIFT)));
  }

  if(! (threadIdx.x % 32)){
    digitCounts[blockIdx.x *32 + (threadIdx.x / 32)] = count[0];
    digitCounts[countSize + (blockIdx.x * 32 + (threadIdx.x / 32))] = count[1];
    digitCounts[2 * countSize  + (blockIdx.x * 32 + (threadIdx.x / 32))] = count[2];

  }
}

  template<uint BIT_SHIFT>
  uint determineDigit(uint *digitCounts,problemInfo_t &info){
  thrust::device_ptr<uint>ptr(digitCounts);
  // printf("SIZE: %d\n", info.curSize);
  // printf("K: %d\n", info.k);
  
  // PrintFunctions::printCudaArray(digitCounts, info.countSize);
  uint sum =0 ;// thrust::reduce(ptr, ptr + info.countSize);
  uint cur[3];
  cur[0] =thrust::reduce(ptr, ptr + info.countSize );
  cur[1] =thrust::reduce(ptr + info.countSize, ptr + (info.countSize * 2));
  cur[2] =thrust::reduce(ptr + (info.countSize * 2), ptr + (info.countSize * 3));

  uint i;
  for(i = 0; i < 3; i++){
    //cur =thrust::reduce(ptr + (info.countSize * i), ptr + (info.countSize * (i + 1)));
    // printf("CUR: %d\n", cur[i]);
    sum += cur[i];
    // printf("SUM: %d\n", sum);
    if(sum >= (info.curSize -info.k) +1){
      info.k -= info.curSize - sum;
      info.curSize = cur[i];
      return i;
    }
  }
  info.curSize -= sum;
  return 3;
 
}

template<uint BIT_SHIFT>
void updateAnswerSoFar(uint &answerSoFar,uint digitValue){
  uint digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}


 template<uint BIT_SHIFT >
 void  digitPass(uint &answerSoFar, uint *digitCounts,problemInfo_t &info){
   
   // cudaMemset(&digitCounts, 0, info.countSize * sizeof(uint) *3);
   //printf("\n\nBIT: %d\n", BIT_SHIFT);
   //uint onesMask = 0x1 << BIT_SHIFT;
   // PrintFunctions::printBinary(answerSoFar);
   // PrintFunctions::printBinary(answerSoFar | onesMask);
   getCounts<BIT_SHIFT><<<info.blocks, info.threadsPerBlock>>>(info.d_vec, info.size,digitCounts,info.totalThreads, answerSoFar, info.countSize);
   info.previousDigit = determineDigit<BIT_SHIFT>(digitCounts,info);
   updateAnswerSoFar<BIT_SHIFT>(answerSoFar,info.previousDigit);
 }


void setupInfo(problemInfo_t &info, uint blocks, uint threadsPerBlock, uint size, uint k, uint *d_vec){
    info.blocks = blocks;
    info.threadsPerBlock = 1024;
    info.totalThreads = info.blocks * info.threadsPerBlock;
    info.size = size;
    info.curSize = size;
    info.k = k;
    info.countSize = info.blocks * (info.threadsPerBlock / 32);
    info.previousDigit = 0;
    info.d_vec = d_vec;
  }



template<typename T>
T runDigitPasses(T *d_vec, uint size, uint k, uint blocks, uint threadsPerBlock){

  uint *digitCounts;
  problemInfo_t info;
  uint answerSoFar = 0;
  setupInfo(info,blocks, threadsPerBlock, size, k,d_vec);
  cudaMalloc(&digitCounts, info.countSize * sizeof(uint)*3);

  // digitPass<31>(answerSoFar,digitCounts, info);
  digitPass<30>(answerSoFar,digitCounts, info);
  // digitPass<29>(answerSoFar,digitCounts, info);
  digitPass<28>(answerSoFar,digitCounts, info);
  // digitPass<27>(answerSoFar,digitCounts, info);
  digitPass<26>(answerSoFar,digitCounts, info);
  // digitPass<25>(answerSoFar,digitCounts, info);
  digitPass<24>(answerSoFar,digitCounts, info);
  // digitPass<23>(answerSoFar,digitCounts, info);
  digitPass<22>(answerSoFar,digitCounts, info);
  // digitPass<21>(answerSoFar,digitCounts, info);
  digitPass<20>(answerSoFar,digitCounts, info);
  // digitPass<19>(answerSoFar,digitCounts, info);
  digitPass<18>(answerSoFar,digitCounts, info);
  // digitPass<17>(answerSoFar,digitCounts, info);
  digitPass<16>(answerSoFar,digitCounts, info);
  // digitPass<15>(answerSoFar,digitCounts, info);
  digitPass<14>(answerSoFar,digitCounts, info);
  // digitPass<13>(answerSoFar,digitCounts, info);
  digitPass<12>(answerSoFar,digitCounts, info);
  // digitPass<11>(answerSoFar,digitCounts, info);
  digitPass<10>(answerSoFar,digitCounts, info);
  // digitPass<9>(answerSoFar,digitCounts, info);
  digitPass<8>(answerSoFar,digitCounts, info);
  // digitPass<7>(answerSoFar,digitCounts, info);
  digitPass<6>(answerSoFar,digitCounts, info);
  // digitPass<5>(answerSoFar,digitCounts, info);
  digitPass<4>(answerSoFar,digitCounts, info);
  // digitPass<3>(answerSoFar,digitCounts, info);
  digitPass<2>(answerSoFar,digitCounts, info);
  // digitPass<1>(answerSoFar,digitCounts, info);
  digitPass<0>(answerSoFar,digitCounts, info);




  return answerSoFar;

}

uint radixSelect(uint *d_vec, uint size, uint k){
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0) ;
  uint blocks = dp.multiProcessorCount;
  uint threadsPerBlock = dp.maxThreadsPerBlock;
  return runDigitPasses(d_vec, size, k, blocks, threadsPerBlock);
}


}
