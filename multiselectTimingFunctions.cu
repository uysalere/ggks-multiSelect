/* Based on timingFunctions.cu */

template <typename T>
 struct results_t{
  float time;
  T * vals;
};

template<typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T **d_vec, T* h_vec, uint size, results_t<T> **result){
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(d_vec, size * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, size * sizeof(T), cudaMemcpyHostToDevice);
  *result = (results_t<T> *)malloc(sizeof(results_t<T>));
}

template<typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T* d_vec, results_t<T> *result, float time){
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
results_t<T>* timeSortAndChooseMultiselect(T *h_vec, uint numElements, uint * kVals, uint kCount){


  T* d_vec;
  T returnValueFromSelect;
  results_t<T> *result;
  float time;
  cudaEvent_t start, stop;
 
  setupForTiming(start,stop, &d_vec, h_vec, numElements, &result);

  thrust::device_ptr<T> dev_ptr(d_vec);
  cudaEventRecord(start, 0);

  thrust::sort(dev_ptr, dev_ptr + numElements);

  cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < kCount; i++)
    result->vals[i] = h_vec[numElements - kVals[i]]; 

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);


  wrapupForTiming(start,stop, d_vec, result, time);
  return result;
}

// FUNCTION TO TIME BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect (T *h_vec, uint numElements, uint * kVals, uint kCount){


  T* d_vec;
  T returnValueFromSelect;
  results_t<T> *result;
  float time;
  cudaEvent_t start, stop;
  cudaDeviceProp dp;

  cudaGetDeviceProperties(&dp,0);


  setupForTiming(start,stop, &d_vec, hostVec, size, &result);

  cudaEventRecord(start, 0);

  BucketSelect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);


  wrapupForTiming(start,stop, deviceVec, result, time);
  return result;
}
