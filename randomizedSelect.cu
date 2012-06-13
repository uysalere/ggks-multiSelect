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

#include "randomizedSelectGenerateSplitters.cu"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_randist.h>
#include <thrust/scan.h>

void choosePivots(int length, int k, int numSplitters, double desiredProbability, int &leftBucket, int &rightBucket){
  double q;
  double p = (double)k/ (double)length;
  int mostLikelyBucket = (double)k / ((double) length/(numSplitters + 1));
  
  q = gsl_ran_binomial_pdf(mostLikelyBucket, p, numSplitters);

  leftBucket = mostLikelyBucket;
  rightBucket = mostLikelyBucket;
  while(q < desiredProbability){
    if(leftBucket > 0){
      leftBucket--;
      q += gsl_ran_binomial_pdf(leftBucket, p, numSplitters);
    }
    if(rightBucket < numSplitters + 1){
      rightBucket++;
      q += gsl_ran_binomial_pdf(rightBucket, p, numSplitters);
    }
  }     
}


template<typename T>
__global__ void queryCount(T *list,T pivot0, T pivot1, int *threadOffsetsFirst, int *threadOffsetsMiddle,int length,int sizeOfOffsets,int elementsPerThread){

  int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  int idx = blockDim.x * blockId + threadIdx.x;  

  int firstCount = 0, middleCount = 0, i;

  if(idx < sizeOfOffsets){
    int startIndex = idx * elementsPerThread;
    for(i = 0 ; startIndex + i < length && i < elementsPerThread;  i++){
      if(list[startIndex + i]  < pivot0){
        firstCount++;
      }
      else if(list[startIndex + i] <= pivot1){
        middleCount++;
      }
    }
    threadOffsetsFirst[idx] = firstCount;
    threadOffsetsMiddle[idx] = middleCount;
  }
}

template<typename T>
__global__ void partition(T* list,T pivot0, T pivot1, int *threadOffsets, T* middle,int sizeOfTotalList, int elementsPerThread){
  int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  int idx = blockDim.x * blockId + threadIdx.x;
  int elementsWrittenByThread = 0;
  int startIndexOriginalList = idx * elementsPerThread;
  int startOffset = threadOffsets[idx];
  T currentElement;

  int i;
 
  for(i = 0; startIndexOriginalList + i < sizeOfTotalList && i < elementsPerThread; i++){
    currentElement = list[startIndexOriginalList + i];
    if(currentElement >= pivot0 && currentElement <= pivot1){
      middle[startOffset + elementsWrittenByThread++] = currentElement;
    }
  }
}


void offsetCalc(int *threadOffsetsFirst, int *threadOffsetsMiddle, int sizeOfThreadOffsets, int &sizeOfFirstBucket, int &sizeOfMiddleBucket){
  thrust::device_ptr<int>ptr_threadOffsetsFirst(threadOffsetsFirst);
  thrust::device_ptr<int> ptr_threadOffsetsMiddle(threadOffsetsMiddle);
  sizeOfFirstBucket = thrust::reduce(ptr_threadOffsetsFirst, ptr_threadOffsetsFirst + sizeOfThreadOffsets);
  thrust::exclusive_scan(ptr_threadOffsetsMiddle, ptr_threadOffsetsMiddle + sizeOfThreadOffsets + 1, ptr_threadOffsetsMiddle);
  cudaMemcpy(&sizeOfMiddleBucket, threadOffsetsMiddle + sizeOfThreadOffsets, sizeof(int), cudaMemcpyDeviceToHost);
}


int determineIfKthInMiddle(int k, int sizeOfFirstBucket, int sizeOfMiddleBucket){
  if(sizeOfFirstBucket >= k){
    return -1;
  }
  if(sizeOfFirstBucket + sizeOfMiddleBucket >= k){
    return 0;
  }
  return 1;
}




void shiftBuckets(int &leftBucket, int &rightBucket, int numBuckets, int bucketContainingKth,int totalNumberOfBuckets){

  //This deals with the special case of when we are only looking at one bucket.
  //When we adjust one of the buckets by numBuckets we usually have to take into account
  //that one of the buckets is already counted.
  if(numBuckets == 1){
    numBuckets = 2;
  }
  if(bucketContainingKth == -1){
    rightBucket = leftBucket;
    leftBucket = max(0,leftBucket - numBuckets +1);
  }
  else{
    leftBucket = rightBucket;
    rightBucket = min(totalNumberOfBuckets-1 , rightBucket +  numBuckets -1);
  }
}


template<typename T>
T randomizedSelect(T *list,int size,int k, double desiredProbability){

  T kthLargestValue, pivot0,pivot1;
  T *splitters, *middleBucket, *pivotPtr0, *pivotPtr1, *shiftedSplitters;
  int *threadOffsetFirst, *threadOffsetMiddle;
  int sizeOfMiddleBucket, sizeOfFirstBucket, pivotIndex0, pivotIndex1, leftBucket, rightBucket;

  k = size - k + 1;
  int indexOfK = k-1 ;

  //these need to be adjusted depending upon the device being used. 
  int threadsPerBlock = 64;
  int elementsPerThread = 8 / (sizeof(T) / sizeof(int)), kthInMiddleBucket = 1;

  int totalBlocks = ceil((double) size / ((double)elementsPerThread * (double)threadsPerBlock));
  int dimOfBlocks = ceil((double) totalBlocks / (double) 65535);
  int blocksPerDim = ceil((double)totalBlocks / (double)dimOfBlocks);
  dim3 numBlocks(blocksPerDim, dimOfBlocks);
  int sizeOfThreadOffset = totalBlocks * threadsPerBlock;
  int numSplitters = (int) 8 * sqrt((double)size);
 
  //allocate everything possible at the begining
  cudaMalloc(&splitters, (numSplitters + 2) * sizeof(T));
  cudaMalloc(&threadOffsetFirst, (totalBlocks *threadsPerBlock  + 1) * sizeof(int));
  cudaMalloc(&threadOffsetMiddle,(totalBlocks * threadsPerBlock + 1) * sizeof(int));

  shiftedSplitters = splitters +1;
  
  //PIVOT CHOICE
  generateSplitters(list, shiftedSplitters, numSplitters, size, sizeOfThreadOffset, numBlocks,threadsPerBlock);
  thrust::device_ptr<T>ptr_splitters(splitters);
  thrust::sort(ptr_splitters, ptr_splitters + numSplitters +2);
  choosePivots(size, k, numSplitters, desiredProbability, leftBucket,rightBucket);

  //CHECK IF PIVOTS PUT KTH INTO MIDDLE BUCKET
  while( kthInMiddleBucket != 0){
    pivotIndex0 = leftBucket - 1;
    pivotIndex1 = rightBucket;
    pivotPtr0 = shiftedSplitters + pivotIndex0;
    pivotPtr1 = shiftedSplitters + pivotIndex1;
    cudaMemcpy(&pivot0, pivotPtr0, sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pivot1, pivotPtr1, sizeof(T), cudaMemcpyDeviceToHost);

    queryCount<<<numBlocks, threadsPerBlock>>>(list,pivot0,pivot1, threadOffsetFirst, threadOffsetMiddle,size, sizeOfThreadOffset, elementsPerThread);

    offsetCalc(threadOffsetFirst, threadOffsetMiddle,sizeOfThreadOffset, sizeOfFirstBucket, sizeOfMiddleBucket); 
    kthInMiddleBucket = determineIfKthInMiddle(k, sizeOfFirstBucket, sizeOfMiddleBucket);
    if( kthInMiddleBucket != 0){
      shiftBuckets(leftBucket, rightBucket,( rightBucket - leftBucket) +1, kthInMiddleBucket, numSplitters+1);
    }
  }
 

  cudaMalloc(&middleBucket,sizeOfMiddleBucket * sizeof(T));
  //GRAB THE ELEMENTS FROM THE MIDDLE BLOCK
  partition<<<numBlocks, threadsPerBlock>>>(list, pivot0, pivot1, threadOffsetMiddle, middleBucket, size, elementsPerThread);

 
  //Sort the middle bucket, then take the kth larges element
  thrust::device_ptr<T>d_middle(middleBucket);
  thrust::sort(d_middle, d_middle + sizeOfMiddleBucket);


  cudaMemcpy(&kthLargestValue, middleBucket + indexOfK -sizeOfFirstBucket, sizeof(T), cudaMemcpyDeviceToHost);

  //free up allocated memory on device
  cudaFree(middleBucket);
  cudaFree(splitters);
  cudaFree(threadOffsetFirst);
  cudaFree(threadOffsetMiddle);

  //return the kthLargestValue
  return kthLargestValue;
}

template<typename T>
T randomizedSelectSelectWrapper(T *list,int size,int k){
  lanlSelect(list,size, k, .90);
}
