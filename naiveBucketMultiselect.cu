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
 *
 * Modified by Erik Opavsky and Emircan Uysaler for multiselection
 */
#include <stdio.h>
#include <thrust/binary_search.h>  
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace NaiveBucketMultiselect{
  using namespace std;

#define MAX_THREADS_PER_BLOCK 1024
#define CUTOFF_POINT 200000 
#define NUM_PIVOTS 17

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

  cudaEvent_t start, stop;
  float time;

  /* start a timer with selection = 0
   * stop a timer with selection = 1
   */
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

  // This function initializes a vector to all zeros on the host (CPU)
  void setToAllZero(uint* deviceVector, int length){
    cudaMemset(deviceVector, 0, length * sizeof(uint));
  }

  // this function assigns elements to buckets
  template <typename T>
  __global__ void assignBucket(T * d_vector, int length, int numBuckets, double slope, T minimum, uint* elementToBucket, uint* bucketCount, int offset) {
  
    int threadIndex = threadIdx.x;  
    int index = blockDim.x * blockIdx.x + threadIndex;
    int bucketIndex;

 
    //variables in shared memory for fast access
    extern __shared__ uint sharedBuckets[];
    __shared__ int sbucketNums;
    __shared__ double sMin;
    sbucketNums = numBuckets;
    sMin = minimum;

    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if (threadIndex < length)    {
      int i;

      for (i = index; i < length; i += offset) {   
        //calculate the bucketIndex for each element
        bucketIndex =  (d_vector[i] - sMin) * slope;

        //if it goes beyond the number of buckets, put it in the last bucket
        if (bucketIndex >= sbucketNums) {
          bucketIndex = sbucketNums - 1;
        }

        elementToBucket[i] = bucketIndex;
        atomicInc(&sharedBuckets[bucketIndex], length);
      }
    }

    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        atomicAdd(&bucketCount[i * MAX_THREADS_PER_BLOCK + threadIndex], sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex]);
  }

  // copy elements from specified buckets in d_vector to newArray
  template <typename T>
  __global__ void copyElements(T* d_vector, int length, uint* elementToBucket, uint * buckets, const int numBuckets, T* newArray, uint * counter, int offset) {
   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint sharedBuckets[];
    if (threadIdx.x < numBuckets)
      sharedBuckets[threadIdx.x] = buckets[threadIdx.x];

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

  // this function finds the buckets containing the kth elements we are looking for (works on the host)
  inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, int numBuckets, uint * kList, int kListCount, uint * sums, uint * kthBuckets) {

    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount, numBuckets * sizeof(uint), cudaMemcpyDeviceToHost));

    int kBucket = 0;
    int k;
    int sum = h_bucketCount[0];

    for(int i = 0; i < kListCount; i++) {
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
      for (int i = 0; i < kListCount; i++) 
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
    int numBuckets = 1024;
    int offset = blocks * threads;    
	
    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket;    //array showing what bucket every element is in
    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    size_t totalBucketSize = numBuckets * sizeof(uint);
    uint * h_bucketCount = (uint*) malloc (totalBucketSize);

    uint * d_bucketCount; //array showing the number of elements in each bucket
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

    CUDA_CALL(cudaMalloc(&d_kIndices, kListCount * sizeof (uint)));
    CUDA_CALL(cudaMalloc(&d_kList, kListCount * sizeof (uint)));

    for (int i = 0; i < kListCount; i++) {
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
    /// ****STEP 4: Generate Slope
    /// ***********************************************************

    //Calculate max-min
    double range = maximum - minimum;
    //Calculate the slope, i.e numBuckets/range
    double slope = (numBuckets - 1) / range;

    /// ***********************************************************
    /// ****STEP 5: Assign elements to buckets
    /// 
    /// ***********************************************************

    //Distribute elements into their respective buckets
    assignBucket<<<numBlocks, threadsPerBlock, numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, slope, minimum, d_elementToBucket, d_bucketCount, offset);

    /// ***********************************************************
    /// ****STEP 6: Find the kth buckets
    /// and their respective update indices
    /// ***********************************************************
    findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kList, kListCount, kthBucketScanner, kthBuckets);


    //we must update K since we have reduced the problem size to elements in the kth bucket
    // get the index of the first element
    kList[0] -= kthBucketScanner[0];
    numUniqueBuckets = 1;
    uniqueBuckets[0] = kthBuckets[0];
    elementsInUniqueBucketsSoFar = 0;

    for (int i = 1; i < kListCount; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        elementsInUniqueBucketsSoFar += h_bucketCount[kthBuckets[i-1]];
        uniqueBuckets[numUniqueBuckets] = kthBuckets[i];
        numUniqueBuckets++;
      }
      kList[i] = elementsInUniqueBucketsSoFar + kList[i] - kthBucketScanner[i];
    }

    //store the length of the newly copied elements
    newInputLength = elementsInUniqueBucketsSoFar + h_bucketCount[kthBuckets[kListCount - 1]];

    printf("naive bucketmultiselect total kbucket_count = %d\n", newInputLength);

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


    copyElements<<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(uint)>>>(d_vector, length, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets, newInput, d_uniqueBucketIndexCounter, offset);

    /*
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
    /*
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
    */
    // sort the vector
    thrust::device_ptr<T>newInput_ptr(newInput);
    thrust::sort(newInput_ptr, newInput_ptr + newInputLength);
    
    for (int i = 0; i < kListCount; i++) {
      CUDA_CALL(cudaMemcpy(output + kIndices[i], newInput + kList[i] - 1, sizeof (T), cudaMemcpyDeviceToHost));
    }

    //free all used memory
    cudaFree(d_elementToBucket);  
    cudaFree(d_bucketCount); 
    cudaFree(newInput); 
    cudaFree(d_uniqueBuckets);
    cudaFree(d_uniqueBucketIndexCounter); 
    free(h_bucketCount);

    return 0;
  }


  /**************************************************************************/
  /**************************************************************************/
  //THIS IS THE BUCKETSELECT FUNCTION WRAPPER THAT CHOOSES THE CORRECT VERSION
  //OF BUCKET SELECT TO RUN BASED ON THE INPUT LENGTH
  /**************************************************************************/
  template <typename T>
  T naiveBucketMultiselectWrapper(T * d_vector, int length, uint * kList_ori, int kListCount, T * outputs, int blocks, int threads) { 

    uint kList[kListCount];
    for(int i = 0; i < kListCount; i++) 
      kList[i] = length - kList_ori[i] + 1;
   
    phaseOne(d_vector, length, kList, kListCount, outputs, blocks, threads);

    return 0;
  }

}
