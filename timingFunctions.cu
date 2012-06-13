/* Copyright 2011 Russel Steinbach, Jeffrey Blanchard, Bradley Gordon,
 *   and Toluwaloju Alabi
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
template <typename T>
 struct results_t{
  float time;
  T val;
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
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T* d_vec, results_t<T> *result, float time, T value){
  cudaFree(d_vec);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->val = value;
  result->time = time;
  //   cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////
//          THE SORT AND CHOOSE TIMING FUNCTION
/////////////////////////////////////////////////////////////////
template<typename T>
results_t<T>* timeSortAndChoose(T *h_vec, uint numElements, uint k){

  
  T* d_vec;
  T returnValueFromSelect;
  results_t<T> *result;
  float time;
  cudaEvent_t start, stop;
 
  setupForTiming(start,stop, &d_vec, h_vec, numElements, &result);

  thrust::device_ptr<T> dev_ptr(d_vec);
  cudaEventRecord(start, 0);

  thrust::sort(dev_ptr, dev_ptr + numElements);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);

  cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
  returnValueFromSelect = h_vec[numElements - k];

  wrapupForTiming(start,stop, d_vec, result, time,returnValueFromSelect);
  return result;
}



/////////////////////////////////////////////////////////////////
//          THE RADIX SELECT TIMING FUNCTION
/////////////////////////////////////////////////////////////////

template<typename T>
results_t<T>* timeRadixSelect(T *h_vec, uint numElements, uint k){


  float time;
  cudaEvent_t start,stop;
  results_t<T> *result;
  T returnValueFromSelect;
  T *d_vec;

  setupForTiming(start,stop, &d_vec, h_vec, numElements, &result);
  thrust::device_ptr<T> dev_ptr(d_vec);

  cudaEventRecord(start,0);
  //CALL THE WRAPPER FUNCTION
  returnValueFromSelect =  RadixSelect::RadixSelectWrapper(d_vec,numElements,k);
    
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);

  wrapupForTiming(start,stop, d_vec, result, time,returnValueFromSelect);

  return result;
}
template<typename T>
results_t<T>* timeInplaceRadixSelect(T *h_vec, uint numElements, uint k){


  float time;
  cudaEvent_t start,stop;
  results_t<T> *result;
  T returnValueFromSelect;
  T *d_vec;

  setupForTiming(start,stop, &d_vec, h_vec, numElements, &result);
  thrust::device_ptr<T> dev_ptr(d_vec);

  cudaEventRecord(start,0);
  //CALL THE WRAPPER FUNCTION
  returnValueFromSelect =  InplaceRadix::inplaceRadixSelectWrapper(d_vec,numElements,k);
    
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);

  wrapupForTiming(start,stop, d_vec, result, time,returnValueFromSelect);

  return result;
}

// /////////////////////////////////////////////////////////////////
// //          THE SLICING TIMING FUNCTION
// /////////////////////////////////////////////////////////////////
template<typename T>
results_t<T>* timeCuttingPlane(T *h_vec, uint size, uint k){
  float time;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  results_t<T> *result;
  T returnValueFromSelect;
  result = (results_t<T> *)malloc(sizeof(results_t<T>));

  thrust::device_vector<T> DataD(size);
  thrust::copy(h_vec,h_vec + size,DataD.begin());

  cudaEventRecord(start,0);

 returnValueFromSelect = cp_select::median_min(DataD.begin(), DataD.end(),  7,  1, 1 + size - k);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->val =returnValueFromSelect;
  result->time = time;

  DataD.resize(0);
  return result;
  
}

//FUNCTION TO TIME BUCKET SELECT
template<typename T>
results_t<T>* timeBucketSelect(T* hostVec, uint size, uint k){
cudaEvent_t start, stop;
  float time;
  results_t<T> *result;
  T retFromSelect;
  T* deviceVec;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);


  setupForTiming(start,stop, &deviceVec, hostVec, size, &result);

  cudaEventRecord(start, 0);

  retFromSelect = BucketSelect::bucketSelectWrapper(deviceVec, size, k, dp.multiProcessorCount, dp.maxThreadsPerBlock);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);


  wrapupForTiming(start,stop, deviceVec, result, time, retFromSelect);
  return result;

}


//FUNCTION TO TIME LANL SELECT
template<typename T>
results_t<T>* timeRandomizedSelect(T* hostVec, uint size, uint k)
{
  cudaEvent_t start, stop;
  float time;
  results_t<T> *result;
  T returnValueFromSelect;
  T* deviceVec;

  setupForTiming(start,stop, &deviceVec, hostVec, size, &result);
 
  cudaEventRecord(start, 0);

  returnValueFromSelect = randomizedSelect(deviceVec,size,k,.90);
 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);


  wrapupForTiming(start,stop, deviceVec, result, time,returnValueFromSelect);
  return result;

}
