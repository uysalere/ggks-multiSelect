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
#include <stdio.h>
#include <thrust/binary_search.h>  
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>

namespace RandomizedBlockedBucketSelect{
  using namespace std;

#define MAX_THREADS_PER_BLOCK 1024
#define CUTOFF_POINT 200000 
#define NUM_PIVOTS 17

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)


  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER CPU FUNCTIONS
  /// ***********************************************************
  /// ***********************************************************

  cudaEvent_t start, stop;
  float time;

  void timing(int selection, int ind){
    if(selection==0) {
      //****//
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start,0);
      //****//
    }
    else {
      //****//
      cudaThreadSynchronize();
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      printf("Time %d: %lf \n", ind, time);
      //****//
    }
  }

  template<typename T>
  void cleanup(uint *h_c, T* d_k, int *etb, uint *bc){
    free(h_c);
    cudaFree(d_k);
    cudaFree(etb);
    cudaFree(bc);
  }

  //This function initializes a vector to all zeros on the host (CPU)
  void setToAllZero(uint* deviceVector, int length){
    cudaMemset(deviceVector, 0, length * sizeof(uint));
  }

  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER GPU FUNCTIONS-KERNELS
  /// ***********************************************************
  /// ***********************************************************

  //this function assigns elements to buckets based off of a randomized sampling of the elements in the vector
  template <typename T>
  __global__ void assignSmartBucket(T * d_vector, int length, int numBuckets, double * slopes, T * pivots, int numPivots, uint* elementToBucket, uint* bucketCount, int offset){
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint bucketIndex;
    int threadIndex = threadIdx.x;  

    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;        
    extern __shared__ uint array[];
    uint * sharedBuckets = (uint *)array;
    double * sharedSlopes = (double *)&sharedBuckets[numBuckets];
    T * sharedPivots = (T *)&sharedSlopes[numPivots-1];
    /*
    uint * sharedBuckets = (uint *)array;
    double * sharedSlopes = (double *)&sharedBuckets[numBuckets];
    T * sharedPivots = (T *)&sharedSlopes[numPivots-1];
    // statically allocating the array gives faster results
    __shared__ double sharedSlopes[NUM_PIVOTS-1];
    __shared__ T sharedPivots[NUM_PIVOTS];
    */
  
    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i <  numBuckets / MAX_THREADS_PER_BLOCK; i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    if(threadIndex < numPivots) {
      *(sharedPivots + threadIndex) = *(pivots + threadIndex);
      if(threadIndex < numPivots-1) {
        sharedSlopes[threadIndex] = slopes[threadIndex];
        if (threadIndex < 1) 
          sharedNumSmallBuckets = numBuckets / (numPivots-1);
      }
    }
    syncthreads();
    
    //assigning elements to buckets and incrementing the bucket counts
    if(index < length) {
      int i;

      for(i = index; i < length; i += offset) {
        T num = d_vector[i];
        int minPivotIndex = 0;
        int maxPivotIndex = numPivots-1;
        int midPivotIndex;

        // find the index of the pivot that is the greatest s.t. lower than or equal to num using binary search
        //while (maxPivotIndex > minPivotIndex+1) {
        for(int j = 1; j < numPivots - 1; j*=2) {
          midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
          if (num >= sharedPivots[midPivotIndex])
            minPivotIndex = midPivotIndex;
          else
            maxPivotIndex = midPivotIndex;
        }

        if (sharedSlopes[minPivotIndex]<0) 
          bucketIndex = (minPivotIndex * sharedNumSmallBuckets) + (threadIndex % sharedNumSmallBuckets);   
        //bucketIndex = (minPivotIndex * sharedNumSmallBuckets);          
        else
          bucketIndex = (minPivotIndex * sharedNumSmallBuckets) + (int) ((num - sharedPivots[minPivotIndex]) * sharedSlopes[minPivotIndex]);

        elementToBucket[i] = bucketIndex;
        atomicInc (sharedBuckets + bucketIndex, length);
      }
    }
    
    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for (int i = 0; i < numBuckets / MAX_THREADS_PER_BLOCK; i++) 
      if (threadIndex < numBuckets) 
        //atomicAdd(bucketCount + blockIdx.x * numBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex, sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex]);
        *(bucketCount + blockIdx.x * numBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex) = *(sharedBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex);
  }

  template <typename T>
  __global__ void assignEdgeMinBucket(T * d_vector, int length, int numBuckets, int numPivots, uint * elementToBucket, uint * bucketCount, int offset, double slope, T lowPivot, T highPivot) {
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint bucketIndex;
    int threadIndex = threadIdx.x;  
    int numSmallBuckets = numBuckets / (numPivots-1);  

    __shared__ double sharedSlope;
    __shared__ T sharedLowPivot;
    __shared__ T sharedHighPivot;
    extern __shared__ uint sharedBuckets[];
    
    if (threadIndex < numSmallBuckets) 
      sharedBuckets[threadIndex] = 0;

    if(threadIndex < 1) {
      sharedLowPivot = lowPivot;
      sharedHighPivot = highPivot;
      sharedSlope = slope;
    }
    
    syncthreads();
    
    //assigning elements to buckets and incrementing the bucket counts
    if(index < length) {
      int i;
      for(i = index; i < length; i += offset) {
        T num = d_vector[i];
        if (num < sharedHighPivot) {
          bucketIndex = (int) ((num - sharedLowPivot) * sharedSlope);
          elementToBucket[i] = bucketIndex;
          atomicInc (sharedBuckets + bucketIndex, length);
        }
      }
    }
        
    syncthreads();

      if (threadIndex < numSmallBuckets)
        *(bucketCount + blockIdx.x * numBuckets + threadIndex) = *(sharedBuckets + threadIndex);
        
  }

  template <typename T>
  __global__ void assignEdgeMaxBucket(T * d_vector, int length, int numBuckets, int numPivots, uint* elementToBucket, uint* bucketCount, int offset, double slope, T lowPivot, T highPivot){
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint bucketIndex;
    int threadIndex = threadIdx.x;  
    int numSmallBuckets = numBuckets / (numPivots-1);  

    __shared__ double sharedSlope;
    __shared__ T sharedLowPivot;
    __shared__ T sharedHighPivot;
    __shared__ int lastBigBucketIndex;
    extern __shared__ uint sharedBuckets[];
    
    if (threadIndex < numSmallBuckets) 
      sharedBuckets[threadIndex] = 0;

    if(threadIndex < 1) {
      sharedLowPivot = lowPivot;
      sharedHighPivot = highPivot;
      sharedSlope = slope;
      lastBigBucketIndex = numBuckets - numSmallBuckets;
    }
    
    syncthreads();
    
    //assigning elements to buckets and incrementing the bucket counts
    if(index < length) {
      int i;
      for(i = index; i < length; i += offset) {
        T num = d_vector[i];
        if (num >= sharedLowPivot) {
          bucketIndex = (int) ((num - sharedLowPivot) * sharedSlope);
          elementToBucket[i] = lastBigBucketIndex + bucketIndex;
          atomicInc (sharedBuckets + bucketIndex, length);
        }
      }
    }
        
    syncthreads();

      if (threadIndex < numSmallBuckets)
        *(bucketCount + blockIdx.x * numBuckets + lastBigBucketIndex + threadIndex) = *(sharedBuckets + threadIndex);
        
  }

  //this function assigns elements to buckets
  template <typename T>
  __global__ void assignBucket(T* d_vector, int length, int bucketNumbers, double slope, double minimum, int* bucket, uint* bucketCount, int offset){
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int bucketIndex;
    extern __shared__ uint sharedBuckets[];
    int index = threadIdx.x;  
 
    //variables in shared memory for fast access
    __shared__ int sbucketNums;
    __shared__ double sMin;
    sbucketNums = bucketNumbers;
    sMin = minimum;

    //reading bucket counts into shared memory where increments will be performed
    for(int i=0; i < (bucketNumbers/1024); i++) 
      if(index < bucketNumbers) 
        sharedBuckets[i*1024+index] = 0;
    syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if(idx < length)    {
      int i;
      for(i=idx; i< length; i+=offset){   
        //calculate the bucketIndex for each element
        bucketIndex =  (d_vector[i] - sMin) * slope;

        //if it goes beyond the number of buckets, put it in the last bucket
        if(bucketIndex >= sbucketNums){
          bucketIndex = sbucketNums - 1;
        }
        bucket[i] = bucketIndex;
        atomicInc(&sharedBuckets[bucketIndex], length);
      }
    }

    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for(int i=0; i < (bucketNumbers/1024); i++) 
      if(index < bucketNumbers) 
        atomicAdd(&bucketCount[i*1024+index], sharedBuckets[i*1024+index]);
  }

  //this function reassigns elements to buckets
  template <typename T>
  __global__ void reassignBucket(T* d_vector, int *bucket, uint *bucketCount, const int bucketNumbers, const int length, const double slope, const double maximum, const double minimum, int offset, int Kbucket){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ uint sharedBuckets[];
    int index = threadIdx.x;
    int bucketIndex;

    //reading bucket counts to shared memory where increments will be performed
    if(index < bucketNumbers){
      sharedBuckets[index] =0;
    }
    syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if (idx < length){
      int i;

      for(i=idx; i<length; i+=offset){
        if(bucket[i] != Kbucket){
          bucket[i] = bucketNumbers+1;
        }
        else{
          //calculate the bucketIndex for each element
          bucketIndex = (d_vector[i] - minimum) * slope;

          //if it goes beyond the number of buckets, put it in the last bucket
          if(bucketIndex >= bucketNumbers){
            bucketIndex = bucketNumbers - 1;
          }
          bucket[i] = bucketIndex;

          atomicInc(&sharedBuckets[bucketIndex], length);
        }
      }
    }

    syncthreads();

    //reading bucket counts from shared memory back to global memory
    if(index < bucketNumbers){
      atomicAdd(&bucketCount[index], sharedBuckets[index]);
    }
  }

  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int FindKBucket(uint *d_counter, uint *h_counter, const int numBuckets, const int k, uint * sum){
    cudaMemcpy(sum, d_counter, sizeof(uint), cudaMemcpyDeviceToHost);
    int Kbucket = 0;
    
    if (*sum<k){
      cudaMemcpy(h_counter, d_counter, numBuckets * sizeof(uint), cudaMemcpyDeviceToHost);
      while ( (*sum<k) & (Kbucket<numBuckets-1)){
        Kbucket++;
        *sum += h_counter[Kbucket];
      }
    }
    else{
      cudaMemcpy(h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost);
    }
  
    return Kbucket;
  }

  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int findKBucket(uint * d_bucketCount, uint * h_bucketCount, int numBuckets, int k, uint * sum, int numBlocks){
    int sumsRowIndex= numBuckets * (numBlocks-1);
    /*
      for(int j=0; j<numBuckets; j++)
      CUDA_CALL(cudaMemcpy(h_bucketCount + j, d_bucketCount + sumsRowIndex + j, sizeof(uint), cudaMemcpyDeviceToHost));
    */
    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex, sizeof(uint) * numBuckets, cudaMemcpyDeviceToHost));

    int kBucket = 0;
    uint scanner = h_bucketCount[0];

    while ((scanner < k) & (kBucket < numBuckets - 1)) {
      kBucket++;
      scanner += h_bucketCount[kBucket];
    }

    *(sum) = scanner - h_bucketCount[kBucket];
    
    return kBucket;
  }

  __global__ void sumCounts(uint * d_bucketCount, const int numBuckets, const int numBlocks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=1; j< numBlocks; j++) 
      d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    
  }

  __global__ void sumEdgeCounts(uint * d_bucketCount, const int numBuckets, const int numBlocks, const int start) {
    int index = start + threadIdx.x;

    for(int j=1; j< numBlocks; j++) 
      d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    
  }


  //copy elements in the kth bucket to a new array
  template <typename T>
  __global__ void copyElements (T* d_vector, int length, uint* elementToBucket, const int bucket, T* newArray, uint offset, uint * d_bucketCount, int numTotalBuckets){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int  threadIndex = threadIdx.x;

    //__shared__ uint sharedBucket = bucket;
    __shared__ uint sharedBucketCount;
    
    if(threadIndex < 1) 
      sharedBucketCount = d_bucketCount[blockIdx.x * numTotalBuckets + bucket];

    syncthreads();
    
    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        if (elementToBucket[i] == bucket) 
          newArray[atomicDec(&sharedBucketCount, length) - 1] = d_vector[i];
      }
    }

  }

  //copy elements in the kth bucket to a new array
  template <typename T>
  __global__ void copyElement(T* d_vector, int length, int* elementToBucket, int bucket, T* newArray, uint* count, int offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length){
      for(int i=idx; i<length; i+=offset)
        //copy elements in the kth bucket to the new array
        if(elementToBucket[i] == bucket)
          newArray[atomicInc(count, length)] = d_vector[i];
    }

  }

  template <typename T>
  __global__ void GetKvalue(T* d_vector, int * d_bucket, const int Kbucket, const int n, T* Kvalue, int offset )
  {
    uint xIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (xIndex < n) {
      int i;
      for(i=xIndex; i<n; i+=offset){
        if ( d_bucket[i] == Kbucket ) 
          Kvalue[0] = d_vector[i];
      }
    }
  }

  /// ***********************************************************
  /// ***********************************************************
  /// **** GENERATE PIVOTS
  /// ***********************************************************
  /// ***********************************************************

  __host__ __device__
  unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }

  struct RandomNumberFunctor :
    public thrust::unary_function<unsigned int, float> {
    unsigned int mainSeed;

    RandomNumberFunctor(unsigned int _mainSeed) :
      mainSeed(_mainSeed) {}
  
    __host__ __device__
    float operator()(unsigned int threadIdx)
    {
      unsigned int seed = hash(threadIdx) * mainSeed;

      thrust::default_random_engine rng(seed);
      rng.discard(threadIdx);
      thrust::uniform_real_distribution<float> u(0, 1);

      return u(rng);
    }
  };

  template <typename T>
  void createRandomVector(T * d_vec, int size) {
    timeval t1;
    uint seed;

    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
  
    thrust::device_ptr<T> d_ptr(d_vec);
    thrust::transform (thrust::counting_iterator<uint>(0),thrust::counting_iterator<uint>(size), d_ptr, RandomNumberFunctor(seed));
  }

  template <typename T>
  __global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
    *(in + blockIdx.x*blockDim.x + threadIdx.x) = *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
  }


  __global__ void enlargeIndexAndGetElements (float * in, uint * out, uint * list, int size) {
    *(out + blockIdx.x * blockDim.x + threadIdx.x) = (uint) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
  }
  
  template <typename T>
  void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, int totalSmallBuckets) {
    T * d_randoms;
    int endOffset = 0;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);
  
    createRandomVector (d_randoms, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK), MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

    pivots[0] = 0;
    pivots[numPivots - 1] = 0;
    slopes[0] = -1;
    slopes[numPivots - 2] = -1;

    thrust::device_ptr<T>randoms_ptr(d_randoms);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are endOffset away from the min and max pivots
    cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);

    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
      slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets / (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
  
    cudaFree(d_randoms);
  }


  /************************************************************************/
  /************************************************************************/
  //THIS IS THE PHASE TWO FUNCTION WHICH WILL BE CALLED IF THE INPUT
  //LENGTH IS LESS THAN THE CUTOFF OF 2MILLION 200 THOUSAND
  /************************************************************************/


  template <typename T>
  T phaseTwo(T* d_vector, int length, int K, int blocks, int threads, double maxValue = 0, double minValue = 0){ 
    //declaring and initializing variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int numBuckets = 1024;
    int offset = blocks * threads;

    uint sum=0, Kbucket=0, iter=0;
    int Kbucket_count = 0;
 
    //initializing variables for kernel launches
    if(length < 1024){
      numBlocks = 1;
    }
    //variable to store the end result
    T kthValue =0;

    //declaring and initializing other variables
    size_t size = length * sizeof(int);
    size_t totalBucketSize = numBuckets * sizeof(uint);

    //allocate memory to store bucket assignments and to count elements in buckets
    int* elementToBucket;
    uint* d_bucketCount;
    cudaMalloc(&elementToBucket, size);
    cudaMalloc(&d_bucketCount, totalBucketSize);
    uint * h_bucketCount = (uint*)malloc(totalBucketSize);

    T* d_Kth_val;
    cudaMalloc(&d_Kth_val, sizeof(T));

    thrust::device_ptr<T>dev_ptr(d_vector);
    //if max == min, then we know that it must not have had the values passed in. 
    if(maxValue == minValue){
      thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);
      minValue = *result.first;
      maxValue = *result.second;
    }
    double slope = (numBuckets - 1)/(maxValue - minValue);
    //first check is max is equal to min
    if(maxValue == minValue){
      cleanup(h_bucketCount, d_Kth_val, elementToBucket,d_bucketCount);
      return maxValue;
    }

    //make all entries of this vector equal to zero
    setToAllZero(d_bucketCount, numBuckets);
    //distribute elements to bucket
    assignBucket<<<numBlocks, threadsPerBlock, numBuckets*sizeof(uint)>>>(d_vector, length, numBuckets, slope, minValue, elementToBucket, d_bucketCount, offset);

    //find the bucket containing the kth element we want
    Kbucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &sum);
    Kbucket_count = h_bucketCount[Kbucket];

    while ( (Kbucket_count > 1) && (iter < 1000)){
      minValue = max(minValue, minValue + Kbucket/slope);
      maxValue = min(maxValue, minValue + 1/slope);

      K = K - sum + Kbucket_count;

      if ( maxValue - minValue > 0.0f ){
        slope = (numBuckets - 1)/(maxValue-minValue);
        setToAllZero(d_bucketCount, numBuckets);
        reassignBucket<<< numBlocks, threadsPerBlock, numBuckets * sizeof(uint) >>>(d_vector, elementToBucket, d_bucketCount, numBuckets,length, slope, maxValue, minValue, offset, Kbucket);

        sum = 0;
        Kbucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &sum);
        Kbucket_count = h_bucketCount[Kbucket];

        iter++;
      }
      else{
        //if the max and min are the same, then we are done
        cleanup(h_bucketCount, d_Kth_val, elementToBucket, d_bucketCount);
        return maxValue;
      }
    }

    GetKvalue<<<numBlocks, threadsPerBlock >>>(d_vector, elementToBucket, Kbucket, length, d_Kth_val, offset);
    cudaMemcpy(&kthValue, d_Kth_val, sizeof(T), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
  

    cleanup(h_bucketCount, d_Kth_val, elementToBucket, d_bucketCount);
    return kthValue;
  }



  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOne(T* d_vector, int length, int K, int blocks, int threads, int pass = 0){
    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int numBuckets = 1024;
    int offset = blocks * threads;
    int kthBucket, kthBucketCount;
    int newInputLength;
    int* elementToBucket; //array showing what bucket every element is in
    //declaring and initializing other variables

    uint *d_bucketCount, *count; //array showing the number of elements in each bucket
    uint kthBucketScanner = 0;

    size_t size = length * sizeof(int);

    //variable to store the end result
    T kthValue = 0;
    T* newInput;

    //find max and min with thrust
    double maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if(maximum == minimum){
      return maximum;
    }
    //if we want the max or min just return it
    if(K == 1){
      return minimum;
    }
    if(K == length){
      return maximum;
    }		
    //Allocate memory to store bucket assignments
  
    CUDA_CALL(cudaMalloc(&elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBuckets * sizeof(uint);
    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));
    uint* h_bucketCount = (uint*)malloc(totalBucketSize);

    //Calculate max-min
    double range = maximum - minimum;
    //Calculate the slope, i.e numBuckets/range
    double slope = (numBuckets - 1)/range;

    cudaMalloc(&count, sizeof(uint));
    //Set the bucket count vector to all zeros
    setToAllZero(d_bucketCount, numBuckets);

    //Distribute elements into their respective buckets
    assignBucket<<<numBlocks, threadsPerBlock, numBuckets*sizeof(uint)>>>(d_vector, length, numBuckets, slope, minimum, elementToBucket, d_bucketCount, offset);

    kthBucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &kthBucketScanner);
    kthBucketCount = h_bucketCount[kthBucket];
 
    printf("original kthBucketCount = %d\n", kthBucketCount);

    //we must update K since we have reduced the problem size to elements in the kth bucket
    if(kthBucket != 0){
      K = kthBucketCount - (kthBucketScanner - K);
    }

    //copy elements in the kth bucket to a new array
    cudaMalloc(&newInput, kthBucketCount * sizeof(T));
    setToAllZero(count, 1);
    copyElement<<<numBlocks, threadsPerBlock>>>(d_vector, length, elementToBucket, kthBucket, newInput, count, offset);


    //store the length of the newly copied elements
    newInputLength = kthBucketCount;


    //if we only copied one element, then we are done
    if(newInputLength == 1){
      thrust::device_ptr<T>new_ptr(newInput);
      kthValue = new_ptr[0];
      
      //free all used memory
      cudaFree(elementToBucket); cudaFree(d_bucketCount); cudaFree(count); cudaFree(newInput);
      return kthValue;
    }
 
    /*********************************************************************/
    //END OF FIRST PASS, NOW WE PROCEED TO SUBSEQUENT PASSES
    /*********************************************************************/

    //if the new length is greater than the CUTOFF, run the regular phaseOne again
    if(newInputLength > CUTOFF_POINT && pass < 1){
      if(pass > 0){
        cudaFree(d_vector);
      }
      cudaFree(elementToBucket);  cudaFree(d_bucketCount); cudaFree(count);
      kthValue = phaseOne(newInput, newInputLength, K, blocks, threads,pass + 1);
    }
    else{
      minimum = max(minimum, minimum + kthBucket/slope);
      maximum = min(maximum, minimum + 1/slope);
      kthValue = phaseTwo(newInput,newInputLength, K, blocks, threads,maximum, minimum);
    }
    

    //free all used memory
    cudaFree(elementToBucket);  cudaFree(d_bucketCount); cudaFree(newInput); cudaFree(count);

    return kthValue;
  }

  
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBLOCKEDBUCKETSELECT ************************/
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBLOCKEDBUCKETSELECT ************************/
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBLOCKEDBUCKETSELECT ************************/
 

  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOneR(T* d_vector, int length, int K, int blocks, int threads, int pass = 0){
    /// ***********************************************************
    /// ****STEP 1: Find Min and Max of the whole vector
    /// ****We don't need to go through the rest of the algorithm if it's flat
    /// ***********************************************************
    /*
    timing(0, 1);
    T maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if(maximum == minimum){
      return maximum;
    }
    //if we want the max or min just return it
    if(K == 1){
      return minimum;
    }
    if(K == length){
      return maximum;
    }		
    timing(1, 1);
    */
    /// ***********************************************************
    /// ****STEP 2: Declare variables and allocate memory
    /// **** Declare Variables
    /// ***********************************************************

    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int numBuckets = 4096;
    int offset = blocks * threads;

    // variables for the randomized selection
    int numPivots = NUM_PIVOTS;
    int sampleSize = MAX_THREADS_PER_BLOCK;
    int numSmallBuckets = numBuckets / (numPivots-1);

    // pivot variables
    double slopes[numPivots - 1];
    double * d_slopes; 
    T pivots[numPivots];
    T * d_pivots;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint* d_elementToBucket; //array showing what bucket every element is in
    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBlocks * numBuckets * sizeof(uint);
    uint h_bucketCount[numBuckets]; //array showing the number of elements in each bucket
    uint * d_bucketCount; 

    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));

    // bucket counters
    int kthBucket;
    uint kthBucketScanner = 0;

    // variable to store the end result
    int newInputLength;
    T* newInput;
    T kthValue = 0;

    /// ***********************************************************
    /// ****STEP 3: Generate Pivots and Slopes
    /// Declare slopes and pivots
    /// ***********************************************************

    CUDA_CALL(cudaMalloc(&d_slopes, (numPivots - 1) * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

    //Find bucket sizes using a randomized selection
    generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize, numBuckets);

    // make any slopes that were infinity due to division by zero (due to no 
    //  difference between the two associated pivots) into zero, so all the
    //  values which use that slope are projected into a single bucket
    for (register int i = 0; i < numPivots - 1; i++)
      if (isinf(slopes[i]))
        slopes[i] = 0;
    
    CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots * sizeof(T), cudaMemcpyHostToDevice));

    /// ***********************************************************
    /// ****STEP 4: Assign elements to buckets
    /// 
    /// ***********************************************************

    //Distribute elements into their respective buckets
    assignSmartBucket<T><<<numBlocks, threadsPerBlock, numPivots * sizeof(T) + (numPivots-1) * sizeof(double) + numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, d_slopes, d_pivots, numPivots, d_elementToBucket, d_bucketCount, offset);

    sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

    /// ***********************************************************
    /// ****STEP 5: Find the kth buckets
    /// and their respective update indices
    /// ***********************************************************

    kthBucket = findKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &kthBucketScanner, numBlocks);

    if (kthBucket < numSmallBuckets) {
      /*
      thrust::device_ptr<T>dev_ptr(d_vector);
      thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);
      */
      thrust::device_ptr<T>dev_ptr(d_vector);

      pivots[0] = *(thrust::min_element(dev_ptr, dev_ptr + length));   
      slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);
 
      assignEdgeMinBucket<T><<<numBlocks, threadsPerBlock,  numSmallBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, numPivots, d_elementToBucket, d_bucketCount, offset, slopes[0], pivots[0], pivots[1]);

      sumEdgeCounts<<<1, numSmallBuckets>>>(d_bucketCount, numBuckets, numBlocks, 0);

      kthBucket = findKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &kthBucketScanner, numBlocks);   

    } else if (kthBucket > (numBuckets - numSmallBuckets -1)) { 
      thrust::device_ptr<T>dev_ptr(d_vector);

      pivots[numPivots - 1] = *(thrust::max_element(dev_ptr, dev_ptr + length));   
      printf("max %f\n", pivots[numPivots - 1]);   
      slopes[numPivots - 2] = numSmallBuckets / (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);
 
      assignEdgeMaxBucket<T><<<numBlocks, threadsPerBlock,  numSmallBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, numPivots, d_elementToBucket, d_bucketCount, offset, slopes[numPivots - 2], pivots[numPivots - 2], pivots[numPivots - 1]);

      sumEdgeCounts<<<1, numSmallBuckets>>>(d_bucketCount, numBuckets, numBlocks, numBuckets-numSmallBuckets);

      kthBucket = findKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &kthBucketScanner, numBlocks);   
    }

    newInputLength = h_bucketCount[kthBucket];    
    K -= kthBucketScanner;

    printf("kthbucketcount= %d\n", newInputLength);
    printf("blocked randomselect updated k = %d\n", K);

    /// ***********************************************************
    /// ****STEP 6: Copy the kth buckets
    /// only unique ones
    /// ***********************************************************

    // allocate memories
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));

    copyElements<T><<<numBlocks, threadsPerBlock>>>(d_vector, length, d_elementToBucket, kthBucket, newInput, offset, d_bucketCount, numBuckets);

    //if we only copied one element, then we are done
    if(newInputLength == 1){
      thrust::device_ptr<T>new_ptr(newInput);
      kthValue = new_ptr[0];
      
      //free all used memory
      cudaFree(d_bucketCount); 
      cudaFree(d_elementToBucket); 
      cudaFree(d_pivots); 
      cudaFree(d_slopes); 
      cudaFree(newInput); 
      return kthValue;
    }
 
    /*********************************************************************/
    //END OF FIRST PASS, NOW WE PROCEED TO SUBSEQUENT PASSES
    /*********************************************************************/

    //if the new length is greater than the CUTOFF, run the regular phaseOne again
    if(newInputLength > CUTOFF_POINT && pass < 1){
      if(pass > 0){
        cudaFree(d_vector);
      }
      cudaFree(d_bucketCount); 
      cudaFree(d_elementToBucket); 
      cudaFree(d_pivots); 
      cudaFree(d_slopes); 
      kthValue = phaseOne(newInput, newInputLength, K, blocks, threads,pass + 1);
    }
    else{
      // find boundaries of kth bucket
      /*
      int pivotOffset = numBuckets / (numPivots - 1);
      int pivotIndex = kthBucket/pivotOffset;
      int pivotInnerindex = kthBucket - pivotOffset * pivotIndex;
      minimum = max(minimum, (T) (pivots[pivotIndex] + pivotInnerindex / slopes[pivotIndex])); 
      maximum = min(maximum, (T) (pivots[pivotIndex] + (pivotInnerindex+1) / slopes[pivotIndex]));
      */

      if (newInputLength<33000) {
        thrust::device_ptr<T>newInput_ptr(newInput);
        thrust::sort(newInput_ptr, newInput_ptr + newInputLength);
        cudaMemcpy (&kthValue, newInput + K - 1, sizeof (T), cudaMemcpyDeviceToHost);
      } else
        kthValue = phaseTwo(newInput,newInputLength, K, blocks, threads);
      
      /*
      minimum = max(minimum, minimum + kthBucket/slope);
      maximum = min(maximum, minimum + 1/slope);
      kthValue = phaseTwo(newInput,newInputLength, K, blocks, threads,maximum, minimum);
      */
    }

    //free all used memory
    cudaFree(d_elementToBucket); 
    cudaFree(d_bucketCount); 
    cudaFree(d_slopes); 
    cudaFree(d_pivots);
    cudaFree(newInput);


    return kthValue;
  }


  /**************************************************************************/
  /**************************************************************************/
  //THIS IS THE RANDOMIZEDBUCKETSELECT FUNCTION WRAPPER THAT CHOOSES THE CORRECT
  //VERSION OF BUCKET SELECT TO RUN BASED ON THE INPUT LENGTH
  /**************************************************************************/
  template <typename T>
  T randomizedBlockedBucketSelectWrapper(T* d_vector, int length, int K, int blocks, int threads)
  {
    T kthValue;
    //change K to be the kth smallest
    K = length - K + 1;

    if(length <= CUTOFF_POINT)
      {
        kthValue = phaseTwo(d_vector, length, K, blocks, threads);
        return kthValue;
      }
    else
      {
        //printf("Call PhaseOneR in parent function.\n");
        kthValue = phaseOneR(d_vector, length, K, blocks, threads);
        // printf("After Call PhaseOneR in parent function, kthvalue = %f.\n", kthValue);
        return kthValue;
      }

  }
}
