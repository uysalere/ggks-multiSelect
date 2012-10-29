/* Based on timingFunctions.cu */
#include <stdlib.h>

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {    \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

template <typename T>
 struct results_t {
  float time;
  T * vals;
};

template <typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T * h_vec, T ** d_vec, results_t<T> ** result, uint numElements, uint kCount) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc(d_vec, numElements * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, numElements * sizeof(T), cudaMemcpyHostToDevice);

  *result = (results_t<T> *) malloc (sizeof (results_t<T>));
  (*result)->vals = (T *) malloc (kCount * sizeof (T));
}

template <typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, float time, results_t<T> * result) {
  result->time = time;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //   cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////
//          THE SORT AND CHOOSE TIMING FUNCTION
/////////////////////////////////////////////////////////////////


template <typename T>
__global__ void copyInChunk(T * outputVector, T * inputVector, uint * kList, uint kListCount, uint numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < kListCount) 
    *(outputVector + idx) = *(inputVector + (numElements - (*(kList + idx))));
  
}

template<typename T>
results_t<T>* timeSortAndChooseMultiselect(T *h_vec, uint numElements, uint * kVals, uint kCount, uint * mainSeed) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);
  thrust::device_ptr<T> dev_ptr(d_vec);
  thrust::sort(dev_ptr, dev_ptr + numElements);

  /*
  for (int i = 0; i < kCount; i++)
    cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);
  */

  T * d_output;
  uint * d_kList;

  cudaMalloc (&d_output, kCount * sizeof (T));
  cudaMalloc (&d_kList, kCount * sizeof(uint));
  cudaMemcpy (d_kList, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice);

  int threads = MAX_THREADS_PER_BLOCK;
  if (kCount < threads)
    threads = kCount;
  int blocks = (int) ceil (kCount / (float) threads);

  copyInChunk<T><<<blocks, threads>>>(d_output, d_vec, d_kList, kCount, numElements);
  cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_kList); 
   
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}

// FUNCTION TO TIME BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount, uint * mainSeed) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, 0);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock, mainSeed);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}

// FUNCTION TO TIME NAIVE BUCKET MULTISELECT
template<typename T>
results_t<T>* timeNaiveBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount, uint * mainSeed) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, 0);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);
  thrust::device_ptr<T> dev_ptr(d_vec);
  thrust::sort(dev_ptr, dev_ptr + numElements);

  
  for (int i = 0; i < kCount; i++)
    cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);
  

  // cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  // NaiveBucketMultiselect::naiveBucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}
