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
 * Modified by Erik Opavsky for multiselection
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

  //this function assigns elements to buckets
  template <typename T>
  __global__ void assignBucket(T * d_vector, int length, int bucketNumbers, double slope, double minimum, int* bucket, uint* bucketCount, int offset){
  
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
  /*
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
  */
  //copy elements in the kth bucket to a new array
  template <typename T>
  __global__ void copyElement(T* d_vector, int length, int* elementToBucket, int * buckets, const int numBuckets, T* newArray, uint* count, int offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length) {
      for(int i=idx; i<length; i+=offset)
        //copy elements in the kth bucket to the new array
        for (int j = 0; j < numBuckets; j++)
          if(elementToBucket[i] == buckets[j])
            newArray[atomicInc(count, length)] = d_vector[i];
    }

  }

  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, const int numBuckets, uint * kVals, int kCount, int * sums, int * kthBuckets){
    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount, numBuckets * sizeof(uint), cudaMemcpyDeviceToHost));
    // printf("h_bucketCount[0] = %u\n", h_bucketCount[0]);
    // printf("h_bucketCount[10] = %u\n", h_bucketCount[10]);
    // printf("h_bucketCount[%d] = %u\n", numBuckets -1, h_bucketCount[numBuckets-1]);
    int kBucket = 0;
    int k;
    int sum=h_bucketCount[0];

    for(int i = 0; i < kCount; i++) {
      k = kVals[i];
      while ((sum < k) & (kBucket < numBuckets-1)) {
        kBucket++; 
        sum += h_bucketCount[kBucket];
      }
      kthBuckets[i]=kBucket;
      sums[i]=sum - h_bucketCount[kBucket];
    }
    return 0;
  }

  /*
  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int FindSmartKBucket(uint *d_counter, uint *h_counter, const int num_buckets,  int k, uint * sum){
    cudaMemcpy(sum, d_counter, sizeof(uint), cudaMemcpyDeviceToHost);
    int Kbucket = 0;
    int warp_size = 32;


    if (*sum<k){
      while ( (*sum<k) & (Kbucket<num_buckets-1)) {
        Kbucket++; 
        if (!((Kbucket-1)%32))
          cudaMemcpy(h_counter + Kbucket, d_counter + Kbucket, warp_size * sizeof(uint), cudaMemcpyDeviceToHost);
        *sum += h_counter[Kbucket];
      }
    }
    else{
      cudaMemcpy(h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost);
    }
  
    return Kbucket;
  }
  */

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


  /************************************************************************/
  /************************************************************************/
  //THIS IS THE PHASE TWO FUNCTION WHICH WILL BE CALLED IF THE INPUT
  //LENGTH IS LESS THAN THE CUTOFF OF 2MILLION 200 THOUSAND
  /************************************************************************/
  /*

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
  */


  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOne(T* d_vector, int length, uint * kVals, int kCount, T * output, int blocks, int threads, int pass = 0){
    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int numBuckets = 1024;
    int offset = blocks * threads;

    // bucket counters
    int kthBuckets[kCount], kthBucketScanner[kCount], kIndices[kCount], markedBuckets[kCount], numMarkedBuckets, elementsInBucketsSoFar;
    int * d_markedBuckets;


    for (int i = 0; i < kCount; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = i;
    }

    // variable to store the end result
    int newInputLength;
    T* newInput;

    //find max and min with thrust
    double maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if(maximum == minimum){
      for (int i=0; i<kCount; i++) 
        output[i] = minimum;

      return 0;
    }
    //if we want the max or min just return it
    /*
    if(K == 1){
      return minimum;
    }
    if(K == length){
      return maximum;
      }*/	
	
    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(int);
    uint * count; 

    size_t totalBucketSize = numBuckets * sizeof(uint);
    uint * h_bucketCount = (uint*) malloc (totalBucketSize);
    uint * d_bucketCount; //array showing the number of elements in each bucket
    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));
    int * d_elementToBucket; //array showing what bucket every element is in
    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));


    //Calculate max-min
    double range = maximum - minimum;
    //Calculate the slope, i.e numBuckets/range
    double slope = (numBuckets - 1) / range;


   //Set the bucket count vector to all zeros
    CUDA_CALL(cudaMalloc(&count, sizeof(uint)));
    setToAllZero(d_bucketCount, numBuckets);

    int * d_kIndices;
    uint * d_kVals;
    CUDA_CALL(cudaMalloc(&d_kIndices, kCount * sizeof (int)));
    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, kCount * sizeof (int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_kVals, kCount * sizeof (uint)));
    CUDA_CALL(cudaMemcpy(d_kVals, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice)); 


    // sort the given indices
    thrust::device_ptr<uint>kVals_ptr(d_kVals);
    thrust::device_ptr<int>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kVals_ptr, kVals_ptr + kCount, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, kCount * sizeof (int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kVals, d_kVals, kCount * sizeof (uint), cudaMemcpyDeviceToHost)); 

    //Distribute elements into their respective buckets
    assignBucket<<<numBlocks, threadsPerBlock, numBuckets*sizeof(uint)>>>(d_vector, length, numBuckets, slope, minimum, d_elementToBucket, d_bucketCount, offset);

    cudaThreadSynchronize();

    findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, kCount, kthBucketScanner, kthBuckets);


    //we must update K since we have reduced the problem size to elements in the kth bucket
    // get the index of the first element
    // add the number of elements
    kVals[0] = kVals[0] - kthBucketScanner[0];
    elementsInBucketsSoFar = 0;
    numMarkedBuckets = 1;
    markedBuckets[0] = kthBuckets[0];


    //printf("randomselect total kbucket_count = %u\n", elementsInBucketsSoFar);
    for (int i = 1; i < kCount; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        elementsInBucketsSoFar += h_bucketCount[kthBuckets[i-1]];
        markedBuckets[numMarkedBuckets] = kthBuckets[i];
        numMarkedBuckets++;
      }
      kVals[i] = elementsInBucketsSoFar + kVals[i] - kthBucketScanner[i];
    }


    elementsInBucketsSoFar += h_bucketCount[kthBuckets[kCount-1]];

    //store the length of the newly copied elements
    newInputLength = elementsInBucketsSoFar;


    //copy elements in the kth buckets to a new array
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
    setToAllZero(count, 1);

    CUDA_CALL(cudaMalloc(&d_markedBuckets, numMarkedBuckets * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_markedBuckets, markedBuckets, numMarkedBuckets * sizeof(int), cudaMemcpyHostToDevice));

    copyElement<<<numBlocks, threadsPerBlock>>>(d_vector, length, d_elementToBucket, d_markedBuckets, numMarkedBuckets, newInput, count, offset);

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
    
    for (int i = 0; i < kCount; i++) {
      CUDA_CALL(cudaMemcpy(output + kIndices[i], newInput + kVals[i] - 1, sizeof (T), cudaMemcpyDeviceToHost));
    }

    //free all used memory
    cudaFree(d_elementToBucket);  
    cudaFree(d_bucketCount); 
    cudaFree(newInput); 
    cudaFree(count);
    free(h_bucketCount);

    return 0;
  }


  /**************************************************************************/
  /**************************************************************************/
  //THIS IS THE BUCKETSELECT FUNCTION WRAPPER THAT CHOOSES THE CORRECT VERSION
  //OF BUCKET SELECT TO RUN BASED ON THE INPUT LENGTH
  /**************************************************************************/
  template <typename T>
  T naiveBucketMultiselectWrapper(T * d_vector, int length, uint * kVals_ori, int kCount, T * outputs, int blocks, int threads)
  {

    uint kVals[kCount];
    for(int i=0; i<kCount ; i++)
      kVals[i] = length - kVals_ori[i] + 1;

    //  if(length <= CUTOFF_POINT) 
    //  phaseTwo(d_vector, length, kVals, kCount, outputs, blocks, threads);
    //  else 
    phaseOne(d_vector, length, kVals, kCount, outputs, blocks, threads);
    // void phaseOneR(T* d_vector, int length, uint * kVals, uint kCount, T * outputs, int blocks, int threads, int pass = 0){

    return 0;

  }

}
