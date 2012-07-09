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
      for(i=idx; i< length; i+=offset) {   
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

  //copy elements in the kth bucket to a new array
  template <typename T>
  __global__ void copyElement(T* d_vector, int length, uint* elementToBucket, uint * buckets, const int numBuckets, T* newArray, uint* counter, uint offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint sharedBuckets[];
    if (threadIdx.x <numBuckets)
      sharedBuckets[threadIdx.x]=buckets[threadIdx.x];

    syncthreads();
    int minBucketIndex;
    int maxBucketIndex; 
    int midBucketIndex;
    uint temp;

    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        temp = elementToBucket[i];
        /*
        for(int j=0; j < numBuckets; j++) 
          if (sharedBuckets[j] == temp) 
            newArray[atomicInc(counter, length)] = d_vector[i];
        */
        minBucketIndex = 0;
        maxBucketIndex = numBuckets-1;

        //copy elements in the kth buckets to the new array
        for(int j = 1; j < numBuckets; j*=2) {  
          //while (maxBucketIndex >= minBucketIndex) {  
          midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
          if (temp > sharedBuckets[midBucketIndex])
            minBucketIndex=midBucketIndex+1;
          else
            maxBucketIndex=midBucketIndex;
        }

        if (sharedBuckets[minBucketIndex] == temp) 
          newArray[atomicInc(counter, length)] = d_vector[i];
        
        /*
        if (threadIdx.x <5)
          printf("temp: %d minBucketIndex[%d]:%d maxBucketIndex[%d]:%d\n", temp, minBucketIndex, sharedBuckets[minBucketIndex], maxBucketIndex, sharedBuckets[maxBucketIndex]);
        if ( (temp == sharedBuckets[minBucketIndex]) || (temp == sharedBuckets[maxBucketIndex]) )
          newArray[atomicInc(counter, length)] = d_vector[i];
        else;
        */
    }
    }

  }

  //this function finds the bin containing the kth element we are looking for (works on the host)
  inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, int numBuckets, uint * kVals, int kCount, uint * sums, uint * kthBuckets){
    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount, numBuckets * sizeof(uint), cudaMemcpyDeviceToHost));
    /*
    printf("h_bucketCount[10] = %u\n", h_bucketCount[10]);
    printf("h_bucketCount[%d] = %u\n", numBuckets -1, h_bucketCount[numBuckets-1]);
    */
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
  T phaseTwo(T* d_vector, int length, uint * kVals, uint kCount, T * outputs, int blocks, int threads, double maxValue = 0, double minValue = 0){ 
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
  /*
  template <typename T>
  T phaseOne(T* d_vector, int length, uint * kVals, uint kCount, T * outputs, int blocks, int threads, int pass = 0){
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
    kthBucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, & kthBucketScanner);
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
    

    //free all used memory
    cudaFree(elementToBucket);  cudaFree(d_bucketCount); cudaFree(newInput); cudaFree(count);

    return kthValue;
  }

  */
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBUCKETSELECT ************************/
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBUCKETSELECT ************************/
  /************************* BEGIN FUNCTIONS FOR RANDOMIZEDBUCKETSELECT ************************/
 
  __host__ __device__
  unsigned int hash(unsigned int a)
  {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }

  struct RandomNumberFunctor :
    public thrust::unary_function<unsigned int, float>
  {
    unsigned int mainSeed;

    RandomNumberFunctor(unsigned int _mainSeed) :
      mainSeed(_mainSeed) {}
  
    __host__ __device__
    float operator()(unsigned int threadIdx)
    {
      unsigned int seed = hash(threadIdx) * mainSeed;

      thrust::default_random_engine rng(seed);
      rng.discard(threadIdx);
      thrust::uniform_real_distribution<float> u(0,1);

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
    thrust::transform(thrust::counting_iterator<uint>(0),thrust::counting_iterator<uint>(size),
                      d_ptr, RandomNumberFunctor(seed));
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

    cudaMalloc ((void **) &d_randomFloats, sizeof (float) * sizeOfSample);
  
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

    for (int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1, sizeof (uint), cudaMemcpyDeviceToHost);
      slopes[i-1] = numSmallBuckets / (double) (pivots[i] - pivots[i-1]);
    }

    slopes[numPivots-3] = numSmallBuckets / (double) (pivots[numPivots-2] - pivots[numPivots-3]);
    slopes[numPivots-2] = numSmallBuckets / (double) (pivots[numPivots-1] - pivots[numPivots-2]);
  
    //    for (int i = 0; i < numPivots - 2; i++)
    //  printf("slopes = %lf\n", slopes[i]);

    cudaFree(d_randomInts);
  }
  
  template <typename T>
  void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
      T * d_randoms;
      int endOffset = 22;
      int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
      int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

      cudaMalloc ((void **) &d_randoms, sizeof (T) * sizeOfSample);
  
      createRandomVector (d_randoms, sizeOfSample);

      // converts randoms floats into elements from necessary indices
      enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK), MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

      pivots[0] = min;
      pivots[numPivots-1] = max;

      thrust::device_ptr<T>randoms_ptr(d_randoms);
      thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

      cudaThreadSynchronize();

      // set the pivots which are endOffset away from the min and max pivots
      cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
      cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
      slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

      for (int i = 2; i < numPivots - 2; i++) {
        cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1, sizeof (T), cudaMemcpyDeviceToHost);
        slopes[i-1] = numSmallBuckets / (double) (pivots[i] - pivots[i-1]);
      }

      slopes[numPivots-3] = numSmallBuckets / (double) (pivots[numPivots-2] - pivots[numPivots-3]);
      slopes[numPivots-2] = numSmallBuckets / (double) (pivots[numPivots-1] - pivots[numPivots-2]);
  
      // for (int i = 0; i < numPivots; i++)
      //  printf("pivots = %lf\n", pivots[i]);

      cudaFree(d_randoms);
  }
  
  //this function assigns elements to buckets based off of a randomized sampling of the elements in the vector
  template <typename T>
  __global__ void assignSmartBucket(T * d_vector, int length, int numBuckets, double * slopes, T * pivots, int numPivots, uint* elementToBucket, uint* bucketCount, int offset){
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int bucketIndex;
    int threadIndex = threadIdx.x;  
    
    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;
    sharedNumSmallBuckets = numBuckets / (numPivots-1);

    extern __shared__ uint sharedBuckets[];
    __shared__ double sharedSlopes[NUM_PIVOTS-1];
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
    for(int i=0; i < (numBuckets / 1024); i++) 
      if(threadIndex < numBuckets) 
        sharedBuckets[i*1024+threadIndex] = 0;

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
        atomicInc(sharedBuckets + bucketIndex, length);
      }
    }

    syncthreads();

    //reading bucket counts from shared memory back to global memory
    for(int i=0; i < (numBuckets/1024); i++)
      if(threadIndex < numBuckets)
        atomicAdd(bucketCount + i*1024 + threadIndex, sharedBuckets[i*1024 + threadIndex]);

    /*
   /// Naive while loop implementation

   T num = d_vector[i];
   int j = 1;

   while (num > sharedPivots[j])
   j++;

   int midPivotIndex=j-1;
   if (midPivotIndex >NUM_PIVOTS-1) midPivotIndex = NUM_PIVOTS-1;

   if (threadIndex < 10)
   printf("midPivotIndex = %d\n",midPivotIndex);
    */


    /*
   /// binary search
   T num = d_vector[i];
   int minPivotIndex = 0;
   int maxPivotIndex = numPivots-1;
   int midPivotIndex;

   // find the index of the pivot that is the greatest s.t. lower than or equal to num using binary search
   while (maxPivotIndex >= minPivotIndex) {
   midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
   if (sharedPivots[midPivotIndex+1] <= num)
   minPivotIndex = midPivotIndex+1;
   else if (sharedPivots[midPivotIndex] > num)
   maxPivotIndex = midPivotIndex;
   else
   break;
   }
    */


    /*
   /// temp computation idea
   int j=0;
   T num = d_vector[i];
   int ind = (j+1)*32+(threadIdx.x%32); 
   T temp = num - sharedPivots[ind];
   while ( (j<numPivots-2) && (temp>=0) ) {
   j++;
   temp = num - sharedPivots[(j+1)*32+(threadIdx.x%32)];
   }

   ind = j*32+(threadIdx.x%32); 
   temp = num - sharedPivots[ind];
   bucketIndex = j*sharedNumSmallBuckets + (int)(temp*sharedSlopes[ind]);
   elementToBucket[i] = bucketIndex;
   atomicInc(sharedBuckets + bucketIndex, length);
   }
   }

   syncthreads();

   //reading bucket counts from shared memory back to global memory
   for(int i=0; i < (numBuckets/1024); i++) 
   if(threadIndex < numBuckets) 
   atomicAdd(bucketCount + i*1024 + threadIndex, sharedBuckets[i*1024 + threadIndex]);
    */
  }

  /* this function finds the kth-largest element from the input array */
  template <typename T>
  T phaseOneR(T* d_vector, int length, uint * kList, int kListCount, T * output, int blocks, int threads, int pass = 0){    
    /// ***********************************************************
    /// ****STEP 1: Find Min and Max of the whole vector
    /// ****We don't need to go through the rest of the algorithm if its flat
    /// ***********************************************************

    //timing(0, 1);
    //find max and min with thrust
    T maximum, minimum;

    thrust::device_ptr<T>dev_ptr(d_vector);
    thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);

    minimum = *result.first;
    maximum = *result.second;

    //if the max and the min are the same, then we are done
    if(maximum == minimum) {
      for (int i=0; i<kListCount; i++) 
        output[i] = minimum;
      
      return 0;
    }

    /*
    //if we want the max or min just return it
    if(K == 1){
    return minimum;
    }
    if(K == length){
    return maximum;
    }	
    */	

    //timing(1, 1);
    /// ***********************************************************
    /// ****STEP 2: Declare variables and allocate memory
    /// **** Declare Variables
    /// ***********************************************************

    //timing(0, 2);
    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int numBuckets = 4096;
    int offset = blocks * threads;

    // variables for the randomized selection
    int numPivots = NUM_PIVOTS;
    int sampleSize = MAX_THREADS_PER_BLOCK;

    // pivot vars
    double slopes[numPivots - 1];
    double * d_slopes;
    T pivots[numPivots];
    T * d_pivots;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket; //array showing what bucket every element is in
    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBuckets * sizeof(uint);
    uint h_bucketCount[totalBucketSize]; //array showing the number of elements in each bucket
    uint * d_bucketCount; 
    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));
    setToAllZero(d_bucketCount, numBuckets);

    // array of kth buckets
    int numMarkedBuckets;
    uint * d_kList; 
    uint kthBuckets[kListCount]; 
    uint kthBucketScanner[kListCount]; 
    uint kIndices[kListCount];
    uint * d_kIndices;
    uint markedBuckets[kListCount];
    uint * d_markedBuckets; 
    uint elementsInMarkedBucketsSoFar;
    uint * d_markedBucketIndexCounter;    
    CUDA_CALL(cudaMalloc(&d_kList, kListCount * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_kIndices, kListCount * sizeof (uint)));
    timing(0, 1);
    for (int i=0; i<kListCount; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = i;
    }

    // variable to store the end result
    int newInputLength;
    T* newInput;
    //timing(1, 2);

    /// ***********************************************************
    /// ****STEP 3: Sort the klist
    /// and keep the old index
    /// ***********************************************************

    //timing(0, 3);
    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, kListCount * sizeof (uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kList, kList, kListCount * sizeof (uint), cudaMemcpyHostToDevice)); 

    // sort the given indices
    thrust::device_ptr<uint>kList_ptr(d_kList);
    thrust::device_ptr<uint>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kList_ptr, kList_ptr + kListCount, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, kListCount * sizeof (uint), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kList, d_kList, kListCount * sizeof (uint), cudaMemcpyDeviceToHost)); 

    //timing(1, 3);
    /// ***********************************************************
    /// ****STEP 4: Generate Pivots and Slopes
    /// Declare slopes and pivots
    /// ***********************************************************
    //timing(0, 4);
    CUDA_CALL(cudaMalloc(&d_slopes, (numPivots - 1) * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

    //Find bucket sizes using a randomized selection
    generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize, numBuckets, minimum, maximum);
    
    CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots * sizeof(T), cudaMemcpyHostToDevice));
    //timing(1, 4);

    /// ***********************************************************
    /// ****STEP 5: Assign elements to buckets
    /// 
    /// ***********************************************************

    //timing(0, 5);
    //Distribute elements into their respective buckets
    assignSmartBucket<<<numBlocks, threadsPerBlock, numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, d_slopes, d_pivots, numPivots, d_elementToBucket, d_bucketCount, offset);
    //timing(1, 5);

    /// ***********************************************************
    /// ****STEP 6: Find the kth buckets
    /// and their respective update indices
    /// ***********************************************************
    //timing(0, 6);
    findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kList, kListCount, kthBucketScanner, kthBuckets);
    //timing(1, 6);

    //timing(0, 7);
    //we must update K since we have reduced the problem size to elements in the kth bucket
    // get the index of the first element
    // add the number of elements
    kList[0] = kList[0] - kthBucketScanner[0];
    numMarkedBuckets = 1;
    markedBuckets[0] = kthBuckets[0];
    elementsInMarkedBucketsSoFar=0;
    for (int i = 1; i < kListCount; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        elementsInMarkedBucketsSoFar += h_bucketCount[kthBuckets[i-1]];
        markedBuckets[numMarkedBuckets] = kthBuckets[i];
        numMarkedBuckets++;
      }
      kList[i] = elementsInMarkedBucketsSoFar + kList[i] - kthBucketScanner[i];
    }

    //store the length of the newly copied elements
    newInputLength = elementsInMarkedBucketsSoFar + h_bucketCount[kthBuckets[kListCount-1]];
    //timing(1, 7);
    printf("randomselect total kbucket_count = %d\n", newInputLength);

    /// ***********************************************************
    /// ****STEP 7: Copy the kth buckets
    /// only marked ones
    /// ***********************************************************

    //timing(0, 8);
    // allocate memories
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_markedBuckets, numMarkedBuckets * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_markedBucketIndexCounter, sizeof(uint)));

    //copy marked bucket stuff into device
    CUDA_CALL(cudaMemcpy(d_markedBuckets, markedBuckets, numMarkedBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    setToAllZero(d_markedBucketIndexCounter, 1);
    //timing(1, 8);
    timing(0, 9);

    copyElement<<<numBlocks, threadsPerBlock, numMarkedBuckets * sizeof(uint)>>>(d_vector, length, d_elementToBucket, d_markedBuckets, numMarkedBuckets, newInput, d_markedBucketIndexCounter, offset);
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

    //timing(0, 10);
    // sort the vector
    thrust::device_ptr<T>newInput_ptr(newInput);
    thrust::sort(newInput_ptr, newInput_ptr + newInputLength);
    
    //printf("newInputLength = %d\n", newInputLength);
    for (int i = 0; i < kListCount; i++) {
      //printf("kList[%d] = %u\n", i, kList[i]);
      CUDA_CALL(cudaMemcpy(output + kIndices[i], newInput + kList[i] - 1, sizeof (T), cudaMemcpyDeviceToHost));
    }
    //timing(1, 10);
    
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
  cudaFree(d_kIndices); 
  cudaFree(d_kList); 
  cudaFree(d_markedBuckets); 
  cudaFree(d_markedBucketIndexCounter); 
  cudaFree(d_pivots);

  return 0;

  }

  template <typename T>
  T bucketMultiselectWrapper (T * d_vector, int length, uint * kList_ori, int kListCount, T * outputs, int blocks, int threads) { 
    uint kList[kListCount];
    for(int i=0; i<kListCount ; i++)
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
    phaseOneR(d_vector, length, kList, kListCount, outputs, blocks, threads);
    // void phaseOneR(T* d_vector, int length, uint * kList, uint kListCount, T * outputs, int blocks, int threads, int pass = 0){

    return 0;
  }

}

