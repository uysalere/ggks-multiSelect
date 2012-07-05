/* Based on timingFunctions.cu */
#include <stdlib.h>

template <typename T>
 struct results_t {
  float time;
  T * vals;
};

template<typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T **d_vec, T* h_vec, uint size, results_t<T> **result, uint kCount) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(d_vec, size * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, size * sizeof(T), cudaMemcpyHostToDevice);
  *result = (results_t<T> *) malloc(sizeof(results_t<T>));
}

template<typename T>
void setupForTimingSortAndChoose(cudaEvent_t &start, cudaEvent_t &stop, T **d_vec, T* h_vec, uint size, results_t<T> **result, uint kCount) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(d_vec, size * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, size * sizeof(T), cudaMemcpyHostToDevice);
  *result = (results_t<T> *) malloc(sizeof(results_t<T>));
  (*result)->vals = (T *) malloc (kCount * sizeof (T));
}

template<typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T* d_vec, results_t<T> *result, float time, T * value) {
  cudaFree(d_vec);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->time = time;
  result->vals = value;
  //   cudaDeviceSynchronize();
}

template<typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T* d_vec, results_t<T> *result, float time) {
  cudaFree(d_vec);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->time = time;
  //   cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////
//          THE SORT AND CHOOSE TIMING FUNCTION
/////////////////////////////////////////////////////////////////
template<typename T>
results_t<T>* timeSortAndChooseMultiselect(T *h_vec, uint numElements, uint * kVals, uint kCount) {

  T* d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
 
  setupForTimingSortAndChoose(start, stop, &d_vec, h_vec, numElements, &result, kCount);

  thrust::device_ptr<T> dev_ptr(d_vec);
  cudaEventRecord(start, 0);

  thrust::sort(dev_ptr, dev_ptr + numElements);

  cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < kCount; i++)
    result->vals[i] = h_vec[numElements - kVals[i]];

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);


  wrapupForTiming(start, stop, d_vec, result, time);
  return result;
}

// FUNCTION TO TIME BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount) {


  T* d_vec;
  results_t<T> * result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;
  T* result_vals = (T *) malloc (kCount * sizeof (T));

  cudaGetDeviceProperties(&dp,0);


  setupForTiming(start, stop, &d_vec, h_vec, numElements, &result, kCount);

  cudaEventRecord(start, 0);

  printf("start here\n");
  // void bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads) { 
  BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result_vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  printf("start here\n");
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);


  wrapupForTiming(start, stop, d_vec, result, time, result_vals);
  return result;
}
