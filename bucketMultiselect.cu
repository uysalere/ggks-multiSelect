/* Copyright 2012 Jeffrey Blanchard, Erik Opavsky, and Emircan Uysaler
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
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace BucketMultiselect{
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

  /* start a timer with selection = 0
   * stop a timer with selection = 1
   */
  void timing(int selection, int ind){
    if(selection == 0) {
      //****//
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
      //****//
    }
    else {
      //****//
      cudaThreadSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      printf("Time %d: %lf \n", ind, time);
      //****//
    }
  }

  //This function initializes a vector to all zeros on the host (CPU)
  template<typename T>
  void setToAllZero(T * deviceVector, int length){
    cudaMemset(deviceVector, 0, length * sizeof(T));
  }


  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER GPU FUNCTIONS-KERNELS
  /// ***********************************************************
  /// ***********************************************************

  
  //this function assigns elements to buckets based off of a randomized sampling of the elements in the vector
  template <typename T>
  __global__ void assignSmartBucket (T * d_vector, int length, int numBuckets, double * slopes, T * pivots, int numPivots, uint* elementToBucket, uint* bucketCount, int offset) {
  
    int threadIndex = threadIdx.x;  
    int index = blockDim.x * blockIdx.x + threadIndex;
    int bucketIndex;
    
    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;
    sharedNumSmallBuckets = numBuckets / (numPivots - 1);

    extern __shared__ uint sharedBuckets[];
    __shared__ double sharedSlopes[NUM_PIVOTS - 1];
    __shared__ T sharedPivots[NUM_PIVOTS];

    /*
    //Using one dynamic shared memory for all
    if(threadIndex == 0) {
    //__device__ void func() {
    sharedBuckets = (uint *)array;
    sharedSlopes = (double *) (sharedBuckets + numBuckets);
    sharedPivots = (T *) (sharedSlopes + numPivots-1);
    }*/
  
    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    if (threadIndex < numPivots) {
      sharedPivots[threadIndex] = pivots[threadIndex];
      if (threadIndex < numPivots - 1)
        sharedSlopes[threadIndex] = slopes[threadIndex];
    }

    syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if (index < length) {
      int i;
      for (i = index; i < length; i += offset) {
        T num = d_vector[i];
        int minPivotIndex = 0;
        int maxPivotIndex = numPivots-1;
        int midPivotIndex;

        // binary search tofind the index of the pivot that is the greatest s.t. lower 
        // than or equal to num using binary search
        
        for (int j = 1; j < numPivots - 1; j *= 2) {
          midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
          if (num >= sharedPivots[midPivotIndex])
            minPivotIndex = midPivotIndex;
          else
            maxPivotIndex = midPivotIndex;
        }

        /*
          minPivotIndex = (((num>=sharedPivots[0]) & (num<sharedPivots[1])) * 0) |
          (((num>=sharedPivots[1]) & (num<sharedPivots[2])) * 1) | 
          (((num>=sharedPivots[2]) & (num<sharedPivots[3])) * 2) | 
          (((num>=sharedPivots[3]) & (num<sharedPivots[4])) * 3) | 
          (((num>=sharedPivots[4]) & (num<sharedPivots[5])) * 4) | 
          (((num>=sharedPivots[5]) & (num<sharedPivots[6])) * 5) | 
          (((num>=sharedPivots[6]) & (num<sharedPivots[7])) * 6) | 
          (((num>=sharedPivots[7]) & (num<sharedPivots[8])) * 7) | 
          (((num>=sharedPivots[8]) & (num<sharedPivots[9])) * 8) | 
          (((num>=sharedPivots[9]) & (num<sharedPivots[10])) * 9) | 
          (((num>=sharedPivots[10]) & (num<sharedPivots[11])) * 10) | 
          (((num>=sharedPivots[11]) & (num<sharedPivots[12])) * 11) | 
          (((num>=sharedPivots[12]) & (num<sharedPivots[13])) * 12) | 
          (((num>=sharedPivots[13]) & (num<sharedPivots[14])) * 13) | 
          (((num>=sharedPivots[14]) & (num<sharedPivots[15])) * 14) | 
          (((num>=sharedPivots[15]) & (num<sharedPivots[16])) * 15) | 
          ((num>=sharedPivots[16]) * 16);
        */

        bucketIndex = (minPivotIndex * sharedNumSmallBuckets) + (int) ((num - sharedPivots[minPivotIndex]) * sharedSlopes[minPivotIndex]);
        elementToBucket[i] = bucketIndex;
        atomicInc(sharedBuckets + bucketIndex, length);
      }
    }

    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for(int i = 0; i < (numBuckets/MAX_THREADS_PER_BLOCK); i++)
      if(threadIndex < numBuckets)
        atomicAdd(bucketCount + i * MAX_THREADS_PER_BLOCK + threadIndex, 
                  sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex]);
  }
  
  //this function assigns elements to buckets based off of a randomized sampling of the elements in the vector
  template <typename T>
  __global__ void assignSmartBucketBlocked(T * d_vector, int length, int numBuckets, double * slopes, T * pivots, int numPivots, uint* elementToBucket, uint* bucketCount, int offset){
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int bucketIndex;
    int threadIndex = threadIdx.x;  
    
    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;
    sharedNumSmallBuckets = numBuckets / (numPivots-1);

    extern __shared__ uint sharedBuckets[];
    __shared__ double sharedSlopes[NUM_PIVOTS-1];
    __shared__ T sharedPivots[NUM_PIVOTS];
  
    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    if(threadIndex < numPivots) {
      sharedPivots[threadIndex] = pivots[threadIndex];
      if(threadIndex < numPivots - 1)
        sharedSlopes[threadIndex] = slopes[threadIndex];
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

        bucketIndex = (minPivotIndex * sharedNumSmallBuckets) + (int) ((num - sharedPivots[minPivotIndex]) * sharedSlopes[minPivotIndex]);
        elementToBucket[i] = bucketIndex;
        // hashmap implementation set[bucketindex]=add.i;
        //bucketCount[blockIdx.x * numBuckets + bucketIndex]++;
        atomicInc (sharedBuckets + bucketIndex, length);
      }
    }
    
    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        atomicAdd(bucketCount + blockIdx.x * numBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex, sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex]);
    
  }

  // this function finds the buckets containing the kth elements we are looking for (works on the host)
  inline int findKBuckets (uint * d_bucketCount, uint * h_bucketCount, int numBuckets, uint * kList, int kListCount, uint * sums, uint * kthBuckets) {

    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount, numBuckets * sizeof(uint), cudaMemcpyDeviceToHost));

    int kBucket = 0;
    int k;
    int sum = h_bucketCount[0];

    for (register int i = 0; i < kListCount; i++) {
      k = kList[i];
      while ((sum < k) & (kBucket < numBuckets - 1)) {
        kBucket++; 
        sum += h_bucketCount[kBucket];
      }
      kthBuckets[i] = kBucket;
      sums[i] = sum - h_bucketCount[kBucket];
    }

    return 0;
  }
 

  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int findKBucketsBlocked(uint * d_bucketCount, uint * h_bucketCount, int numBuckets, uint * kVals, int kCount, uint * sums, uint * kthBuckets, int numBlocks){
    int sumsRowIndex= numBuckets * (numBlocks-1);
    /*
      for(int j=0; j<numBuckets; j++)
      CUDA_CALL(cudaMemcpy(h_bucketCount + j, d_bucketCount + sumsRowIndex + j, sizeof(uint), cudaMemcpyDeviceToHost));
    */
    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex, sizeof(uint) * numBuckets, cudaMemcpyDeviceToHost));

    int kBucket = 0;
    int k;
    int sum = h_bucketCount[0];

    for(register int i = 0; i < kCount; i++) {
      k = kVals[i];
      while ((sum < k) & (kBucket < numBuckets - 1)) {
        kBucket++;
        sum += h_bucketCount[kBucket];
      }
      kthBuckets[i] = kBucket;
      sums[i] = sum - h_bucketCount[kBucket];
    }

    return 0;
  }

  // copy elements from specified buckets in d_vector to newArray
  template <typename T>
  __global__ void copyElements(T* d_vector, int length, uint* elementToBucket, uint * buckets, const int numBuckets, T* newArray, uint* counter, int offset) {
 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = numBuckets / MAX_THREADS_PER_BLOCK;

    extern __shared__ uint sharedBuckets[];


    for (int i = 0; i <= loop; i++) {      
      threadIndex = i * blockDim.x + threadIdx.x;
      if(threadIndex < numBuckets) {
        sharedBuckets[threadIndex]=buckets[threadIndex];
      }
    }

    syncthreads();

    // variables for binary search
    int minBucketIndex;
    int maxBucketIndex;
    int midBucketIndex;
    uint tempBucket;

    // go through the whole vector
    if (idx < length) 
      for (int i = idx; i < length; i += offset) {
        tempBucket = elementToBucket[i];
        minBucketIndex = 0;
        maxBucketIndex = numBuckets - 1;

        /* binary search finds whether we wish to copy the current
         * element based on its bucket and the array of buckets
         * specified for copying
         */  
        for (int j = 1; j < numBuckets; j *= 2) {  
          midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;

          if (tempBucket > sharedBuckets[midBucketIndex])
            minBucketIndex = midBucketIndex + 1;
          else
            maxBucketIndex = midBucketIndex;
        }

        if (sharedBuckets[maxBucketIndex] == tempBucket) 
          newArray[atomicInc(counter, length)] = d_vector[i];
      }
  }


  //copy elements in the kth bucket to a new array
  template <typename T>
  __global__ void copyElementsBlocked(T* d_vector, int length, uint* elementToBucket, uint * buckets, const int numBuckets, T* newArray, uint* counter, uint offset, uint * d_bucketCount, int numTotalBuckets){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = numBuckets / MAX_THREADS_PER_BLOCK;

    extern __shared__ uint array[];
    uint * sharedBucketCounts= (uint*)array;
    uint * sharedBuckets= (uint*)&array[numBuckets];

    for (int i = 0; i <= loop; i++) {      
      threadIndex = i * blockDim.x + threadIdx.x;
      if(threadIndex < numBuckets) {
        sharedBuckets[threadIndex]=buckets[threadIndex];
        sharedBucketCounts[threadIndex] = d_bucketCount[blockIdx.x * numTotalBuckets + sharedBuckets[threadIndex]];
      }
    }
    
    syncthreads();

    int minBucketIndex;
    int maxBucketIndex; 
    int midBucketIndex;
    uint temp;
    //uint holder;

    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        temp = elementToBucket[i];
        minBucketIndex = 0;
        maxBucketIndex = numBuckets-1;

        //copy elements in the kth buckets to the new array
        for(int j = 1; j < numBuckets; j*=2) {  
          midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
          if (temp > sharedBuckets[midBucketIndex])
            minBucketIndex=midBucketIndex+1;
          else
            maxBucketIndex=midBucketIndex;
        }

        if (buckets[maxBucketIndex] == temp) 
          //newArray[atomicDec(d_bucketCount + blockIdx.x * numTotalBuckets + temp, length)-1] = d_vector[i];
          newArray[atomicDec(sharedBucketCounts + maxBucketIndex, length)-1] = d_vector[i];
        
      }
    }

  }

  __global__ void sumCounts(uint * d_bucketCount, const int numBuckets, const int numBlocks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=1; j<numBlocks; j++) 
      d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    
  }

  __global__ void reindexCounts(uint * d_bucketCount, const int numBuckets, const int numBlocks, uint * d_reindexCounter, uint * d_markedBuckets, const int numUniqueBuckets) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex<numUniqueBuckets) {
      int index = d_markedBuckets[threadIndex];
      int add = d_reindexCounter[threadIndex];

      for(int j=0; j<numBlocks; j++) 
        d_bucketCount[index + numBuckets*j] += (uint) add;
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
  void generatePivots (uint * pivots, double * slopes, uint * d_list, int sizeOfVector, int numPivots, int sizeOfSample, int totalSmallBuckets, uint min, uint max) {
  
    float * d_randomFloats;
    uint * d_randomInts;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);
  
    d_randomInts = (uint *) d_randomFloats;

    createRandomVector (d_randomFloats, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK), MAX_THREADS_PER_BLOCK>>>(d_randomFloats, d_randomInts, d_list, sizeOfVector);

    pivots[0] = min;
    pivots[numPivots-1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randomInts);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are next to the min and max pivots using the random element endOffset away from the ends
    cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (uint), cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1, sizeof (uint), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1, sizeof (uint), cudaMemcpyDeviceToHost);
      slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets / (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets / (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);

    cudaFree(d_randomFloats);
    cudaFree(d_randomInts);
  }
  
  template <typename T>
  void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
    T * d_randoms;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);
  
    createRandomVector (d_randoms, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK), MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

    pivots[0] = min;
    pivots[numPivots - 1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randoms);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are endOffset away from the min and max pivots
    cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
      slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets / (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets / (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);
  
    cudaFree(d_randoms);
  }

  /// ***********************************************************
  /// ***********************************************************
  /// **** PHASE ONE
  /// ***********************************************************
  /// ***********************************************************

  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOne (T* d_vector, int length, uint * kList, int kListCount, T * output, int blocks, int threads, int pass = 0){    
    /// ***********************************************************
    /// ****STEP 1: Find Min and Max of the whole vector
    /// ****We don't need to go through the rest of the algorithm if it's flat
    /// ***********************************************************

    //find max and min with thrust
    T maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if (maximum == minimum) {
      for (register int i = 0; i < kListCount; i++) 
        output[i] = minimum;
      
      return 0;
    }

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
    int sampleSize = 1024;

    // pivot variables
    double slopes[numPivots - 1];
    double * d_slopes;
    T pivots[numPivots];
    T * d_pivots;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket;    //array showing what bucket every element is in
    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBuckets * sizeof(uint);
    uint * h_bucketCount = (uint *) malloc (totalBucketSize); //array showing the number of elements in each bucket
    uint * d_bucketCount; 
    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));
    setToAllZero(d_bucketCount, numBuckets);

    // array of kth buckets
    int numUniqueBuckets;
    uint * d_kList; 
    uint kthBuckets[kListCount]; 
    uint kthBucketScanner[kListCount]; 
    uint kIndices[kListCount];
    uint * d_kIndices;
    uint uniqueBuckets[kListCount];
    uint * d_uniqueBuckets; 
    uint elementsInUniqueBucketsSoFar;
    uint * d_uniqueBucketIndexCounter;    

    CUDA_CALL(cudaMalloc(&d_kList, kListCount * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_kIndices, kListCount * sizeof (uint)));

    for (register int i = 0; i < kListCount; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = (uint) i;
    }

    // variable to store the end result
    int newInputLength;
    T* newInput;

    /// ***********************************************************
    /// ****STEP 3: Sort the klist
    /// and keep the old index
    /// ***********************************************************

    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, kListCount * sizeof (uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kList, kList, kListCount * sizeof (uint), cudaMemcpyHostToDevice)); 

    // sort the given indices
    thrust::device_ptr<uint>kList_ptr(d_kList);
    thrust::device_ptr<uint>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kList_ptr, kList_ptr + kListCount, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, kListCount * sizeof (uint), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kList, d_kList, kListCount * sizeof (uint), cudaMemcpyDeviceToHost)); 

    cudaFree(d_kIndices); 
    cudaFree(d_kList); 

    /// ***********************************************************
    /// ****STEP 4: Generate Pivots and Slopes
    /// Declare slopes and pivots
    /// ***********************************************************

    CUDA_CALL(cudaMalloc(&d_slopes, (numPivots - 1) * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

    // Find bucket sizes using a randomized selection
    generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize, numBuckets, minimum, maximum);
    
    // make any slopes that were infinity due to division by zero (due to no 
    //  difference between the two associated pivots) into zero, so all the
    //  values which use that slope are projected into a single bucket
    for (register int i = 0; i < numPivots - 1; i++)
      if (isinf(slopes[i]))
        slopes[i] = 0;

    CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots * sizeof(T), cudaMemcpyHostToDevice));

    /// ***********************************************************
    /// ****STEP 5: Assign elements to buckets
    /// 
    /// ***********************************************************

    //Distribute elements into their respective buckets
    assignSmartBucket<<<numBlocks, threadsPerBlock, numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, d_slopes, d_pivots, numPivots, d_elementToBucket, d_bucketCount, offset);

    /// ***********************************************************
    /// ****STEP 6: Find the kth buckets
    /// and their respective update indices
    /// ***********************************************************
    findKBuckets (d_bucketCount, h_bucketCount, numBuckets, kList, kListCount, kthBucketScanner, kthBuckets);

    // we must update K since we have reduced the problem size to elements in the kth bucket
    //  get the index of the first element
    //  add the number of elements
    kList[0] -= kthBucketScanner[0];
    numUniqueBuckets = 1;
    uniqueBuckets[0] = kthBuckets[0];
    elementsInUniqueBucketsSoFar = 0;

    for (register int i = 1; i < kListCount; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        elementsInUniqueBucketsSoFar += h_bucketCount[kthBuckets[i-1]];
        uniqueBuckets[numUniqueBuckets] = kthBuckets[i];
        numUniqueBuckets++;
      }
      kList[i] = elementsInUniqueBucketsSoFar + kList[i] - kthBucketScanner[i];
    }

    //store the length of the newly copied elements
    newInputLength = elementsInUniqueBucketsSoFar + h_bucketCount[kthBuckets[kListCount - 1]];

    printf("bucketmultiselect total kbucket_count = %d\n", newInputLength);

    /// ***********************************************************
    /// ****STEP 7: Copy the kth buckets
    /// only unique ones
    /// ***********************************************************

    // allocate memories
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numUniqueBuckets * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_uniqueBucketIndexCounter, sizeof(uint)));

    //copy unique bucket stuff into device
    CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    setToAllZero(d_uniqueBucketIndexCounter, 1);
   
    timing(0, 9);
 
    copyElements<<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(uint)>>>(d_vector, length, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets, newInput, d_uniqueBucketIndexCounter, offset);
  
    timing(1, 9);

    /// ***********************************************************
    /// ****STEP 8: Sort
    /// and finito
    /// ***********************************************************

    //if we only copied one element, then we are done
    /*
      if(newInputLength == 1){
      thrust::device_ptr<T>new_ptr(newInput);
      kthValue = new_ptr[0];
      
      //free all used memory
      cudaFree(d_elementToBucket); cudaFree(d_bucketCount); cudaFree(count); cudaFree(newInput); cudaFree(d_slopes); cudaFree(d_pivots); free(h_bucketCount);
      return kthValue;
      }*/
 
    /*********************************************************************/
    //END OF FIRST PASS, NOW WE PROCEED TO SUBSEQUENT PASSES
    /*********************************************************************/

    //if the new length is greater than the CUTOFF, run the regular phaseOne again
    /* if(newInputLength > CUTOFF_POINT && pass < 1){
       if(pass > 0){
       cudaFree(d_vector);
       }
       cudaFree(d_elementToBucket); cudaFree(d_bucketCount); cudaFree(count); cudaFree(d_slopes); cudaFree(d_pivots);
       kthValue = phaseOne(newInput, newInputLength, K, blocks, threads,pass + 1);
       }
       else{
       // find boundaries of kth bucket
       int pivotOffset = numBuckets / (numPivots - 1);
       int pivotIndex = kthBucket /pivotOffset;
       int pivotInnerindex = kthBucket - pivotOffset * pivotIndex;
       minimum = max(minimum, (T) (pivots[pivotIndex] + pivotInnerindex / slopes[pivotIndex])); 
       maximum = min(maximum, (T) (pivots[pivotIndex] + (pivotInnerindex+1) / slopes[pivotIndex]));
    */
    //  if (newInputLength<33000) {

    // sort the vector
    thrust::device_ptr<T>newInput_ptr(newInput);
    thrust::sort(newInput_ptr, newInput_ptr + newInputLength);

    //printf("newInputLength = %d\n", newInputLength);
    for (register int i = 0; i < kListCount; i++) 
      CUDA_CALL(cudaMemcpy(output + kIndices[i], newInput + kList[i] - 1, sizeof (T), cudaMemcpyDeviceToHost));
    

    
    /*
      } else
      kthValue = phaseTwo(newInput,newInputLength, K, blocks, threads,maximum, minimum);
      
      /*
      minimum = max(minimum, minimum + kthBucket/slope);
      maximum = min(maximum, minimum + 1/slope);
      kthValue = phaseTwo(newInput,newInputLength, K, blocks, threads,maximum, minimum);
   
      }*/

    //free all used memory
    cudaFree(d_elementToBucket);  
    cudaFree(d_bucketCount); 
    cudaFree(newInput); 
    cudaFree(d_slopes); 
    cudaFree(d_pivots);
    cudaFree(d_uniqueBuckets); 
    cudaFree(d_uniqueBucketIndexCounter); 
    free(h_bucketCount);

    return 0;
  }

  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOneBlocked (T* d_vector, int length, uint * kList, int kListCount, T * output, int blocks, int threads, int pass = 0){    
    /// ***********************************************************
    /// ****STEP 1: Find Min and Max of the whole vector
    /// ****We don't need to go through the rest of the algorithm if it's flat
    /// ***********************************************************

    //find max and min with thrust
    T maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if (maximum == minimum) {
      for (register int i = 0; i < kListCount; i++) 
        output[i] = minimum;
      
      return 0;
    }

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
    int sampleSize = 1024;

    // pivot variables
    double slopes[numPivots - 1];
    double * d_slopes;
    T pivots[numPivots];
    T * d_pivots;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket;    //array showing what bucket every element is in

    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBlocks * numBuckets * sizeof(uint);
    uint h_bucketCount[numBuckets]; //array showing the number of elements in each bucket
    uint * d_bucketCount; 

    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));
    setToAllZero<uint>(d_bucketCount, numBlocks * numBuckets);

    // array of kth buckets
    int numUniqueBuckets;
    uint * d_kList; 
    uint kthBuckets[kListCount]; 
    uint kthBucketScanner[kListCount]; 
    uint kIndices[kListCount];
    uint * d_kIndices;
    uint uniqueBuckets[kListCount];
    uint * d_uniqueBuckets; 
    uint reindexCounter[kListCount];  
    uint * d_reindexCounter;  
    uint * d_uniqueBucketIndexCounter;    

    CUDA_CALL(cudaMalloc(&d_kList, kListCount * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_kIndices, kListCount * sizeof (uint)));

    for (register int i = 0; i < kListCount; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = (uint) i;
    }

    // variable to store the end result
    int newInputLength;
    T* newInput;

    /// ***********************************************************
    /// ****STEP 3: Sort the klist
    /// and keep the old index
    /// ***********************************************************

    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, kListCount * sizeof (uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kList, kList, kListCount * sizeof (uint), cudaMemcpyHostToDevice)); 

    // sort the given indices
    thrust::device_ptr<uint>kList_ptr(d_kList);
    thrust::device_ptr<uint>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kList_ptr, kList_ptr + kListCount, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, kListCount * sizeof (uint), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kList, d_kList, kListCount * sizeof (uint), cudaMemcpyDeviceToHost)); 

    cudaFree(d_kIndices); 
    cudaFree(d_kList); 

    /// ***********************************************************
    /// ****STEP 4: Generate Pivots and Slopes
    /// Declare slopes and pivots
    /// ***********************************************************

    CUDA_CALL(cudaMalloc(&d_slopes, (numPivots - 1) * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

    // Find bucket sizes using a randomized selection
    generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize, numBuckets, minimum, maximum);
    
    // make any slopes that were infinity due to division by zero (due to no 
    //  difference between the two associated pivots) into zero, so all the
    //  values which use that slope are projected into a single bucket
    for (register int i = 0; i < numPivots - 1; i++)
      if (isinf(slopes[i]))
        slopes[i] = 0;

    CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots * sizeof(T), cudaMemcpyHostToDevice));

    /// ***********************************************************
    /// ****STEP 5: Assign elements to buckets
    /// 
    /// ***********************************************************

    //Distribute elements into their respective buckets
    assignSmartBucketBlocked<<<numBlocks, threadsPerBlock, numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, d_slopes, d_pivots, numPivots, d_elementToBucket, d_bucketCount, offset);

    sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

    /// ***********************************************************
    /// ****STEP 6: Find the kth buckets
    /// and their respective update indices
    /// ***********************************************************
    findKBucketsBlocked(d_bucketCount, h_bucketCount, numBuckets, kList, kListCount, kthBucketScanner, kthBuckets, numBlocks);

    // we must update K since we have reduced the problem size to elements in the kth bucket
    //  get the index of the first element
    //  add the number of elements
    uniqueBuckets[0] = kthBuckets[0];
    reindexCounter[0] = 0;
    numUniqueBuckets = 1;
    kList[0] -= kthBucketScanner[0];

    for (int i = 1; i < kListCount; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        uniqueBuckets[numUniqueBuckets] = kthBuckets[i];
        reindexCounter[numUniqueBuckets] = reindexCounter[numUniqueBuckets-1] + h_bucketCount[kthBuckets[i-1]];
        numUniqueBuckets++;
      }
      kList[i] = reindexCounter[numUniqueBuckets-1] + kList[i] - kthBucketScanner[i];
    }

    newInputLength = reindexCounter[numUniqueBuckets-1] + h_bucketCount[kthBuckets[kListCount - 1]];

    printf("bucketmultiselectBlocked total kbucket_count = %d\n", newInputLength);
    printf("numMarkedBuckets = %d\n", numUniqueBuckets);

    CUDA_CALL(cudaMalloc(&d_reindexCounter, numUniqueBuckets * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numUniqueBuckets * sizeof(uint)));

    CUDA_CALL(cudaMemcpy(d_reindexCounter, reindexCounter, numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));

    reindexCounts<<<ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks, d_reindexCounter, d_uniqueBuckets, numUniqueBuckets);

    /// ***********************************************************
    /// ****STEP 7: Copy the kth buckets
    /// only unique ones
    /// ***********************************************************

    // allocate memories
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_uniqueBucketIndexCounter, sizeof(uint)));

    //copy unique bucket stuff into device
    setToAllZero<uint>(d_uniqueBucketIndexCounter, 1);
   
    timing(0, 9);
 
    //copyElements<<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(uint)>>>(d_vector, length, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets, newInput, d_uniqueBucketIndexCounter, offset);
    copyElementsBlocked<<<numBlocks, threadsPerBlock, numUniqueBuckets * 2 * sizeof(uint)>>>(d_vector, length, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets, newInput, d_uniqueBucketIndexCounter, offset, d_bucketCount, numBuckets);
  
    timing(1, 9);

    /// ***********************************************************
    /// ****STEP 8: Sort
    /// and finito
    /// ***********************************************************

    // sort the vector
    thrust::device_ptr<T>newInput_ptr(newInput);
    thrust::sort(newInput_ptr, newInput_ptr + newInputLength);

    //printf("newInputLength = %d\n", newInputLength);
    for (register int i = 0; i < kListCount; i++) 
      CUDA_CALL(cudaMemcpy(output + kIndices[i], newInput + kList[i] - 1, sizeof (T), cudaMemcpyDeviceToHost));

    //free all used memory
    cudaFree(d_pivots);
    cudaFree(d_slopes); 

    cudaFree(d_elementToBucket);  
    cudaFree(d_bucketCount); 
    cudaFree(d_uniqueBuckets); 
    cudaFree(d_uniqueBucketIndexCounter); 
    cudaFree(d_reindexCounter);  

    cudaFree(newInput); 

    return 0;
  }

  template <typename T>
  T bucketMultiselectWrapper (T * d_vector, int length, uint * kList_ori, int kListCount, T * outputs, int blocks, int threads) { 

    uint kList[kListCount];
    for (register int i = 0; i < kListCount; i++) 
      kList[i] = length - kList_ori[i] + 1;
   
    /*
      printf("start here\n");
      printf("k-length: %d\n", length);
      for(int i=0; i<kListCount ; i++)
      printf("k-%d: %d\n", i, kList[i]);
    */

    //  if(length <= CUTOFF_POINT) 
    //  phaseTwo(d_vector, length, kList, kListCount, outputs, blocks, threads);
    //  else 
    phaseOne (d_vector, length, kList, kListCount, outputs, blocks, threads);
    // void phaseOneR(T* d_vector, int length, uint * kList, uint kListCount, T * outputs, int blocks, int threads, int pass = 0){

    return 0;
  }

  template <typename T>
  T bucketMultiselectBlockedWrapper (T * d_vector, int length, uint * kList_ori, int kListCount, T * outputs, int blocks, int threads) { 

    uint kList[kListCount];
    for (register int i = 0; i < kListCount; i++) 
      kList[i] = length - kList_ori[i] + 1;
   
    /*
      printf("start here\n");
      printf("k-length: %d\n", length);
      for(int i=0; i<kListCount ; i++)
      printf("k-%d: %d\n", i, kList[i]);
    */

    //  if(length <= CUTOFF_POINT) 
    //  phaseTwo(d_vector, length, kList, kListCount, outputs, blocks, threads);
    //  else 
    phaseOneBlocked (d_vector, length, kList, kListCount, outputs, blocks, threads);
    // void phaseOneR(T* d_vector, int length, uint * kList, uint kListCount, T * outputs, int blocks, int threads, int pass = 0){

    return 0;
  }
}

