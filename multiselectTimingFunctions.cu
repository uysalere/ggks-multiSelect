/* Based on timingFunctions.cu */
#include <stdlib.h>

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
template<typename T>
results_t<T>* timeSortAndChooseMultiselect(T *h_vec, uint numElements, uint * kVals, uint kCount) {

  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

  cudaEventRecord(start, 0);

  thrust::device_ptr<T> dev_ptr(d_vec);
  thrust::sort(dev_ptr, dev_ptr + numElements);

  for (int i = 0; i < kCount; i++)
    cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}

// FUNCTION TO TIME BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount) {
  T * d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp, 0);

  setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
 
  cudaEventRecord(start, 0);

  // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
  BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  wrapupForTiming(start, stop, time, result);
  cudaFree(d_vec);
  return result;
}
