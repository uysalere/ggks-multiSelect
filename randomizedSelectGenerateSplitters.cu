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

#include <cuda_runtime_api.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <iomanip>
#include <math_constants.h>

//the folowing comes from thrust examples, monte_carlo.cu
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
    public thrust::unary_function<unsigned int, int>
{
    unsigned int mainSeed;
     int sizeOfList;
    RandomNumberFunctor(unsigned int _mainSeed) :
        mainSeed(_mainSeed) {}

    __host__ __device__
        int operator()(unsigned int threadIdx)
  {
    unsigned int seed = hash(threadIdx) * mainSeed;

    thrust::default_random_engine rng(seed);
    rng.discard(threadIdx);
    thrust::uniform_int_distribution<int> u(0,1<<31);

    return u(rng);
  }
};

void generateIndices(uint * d_vec, int size, int sizeOfList){
  timeval t1;
  uint seed;

  gettimeofday(&t1, NULL);
  seed = t1.tv_usec * t1.tv_sec;
  RandomNumberFunctor r(seed);
  r.sizeOfList = sizeOfList;
  
  thrust::device_ptr<uint> d_ptr(d_vec);
  thrust::transform(thrust::counting_iterator<uint>(0),thrust::counting_iterator<uint>(size),
                    d_ptr, RandomNumberFunctor(seed));
}

template<typename T>
__global__ void copyTheSplitters(T* list, T* splitters, uint *splittersIndex, int numSplitters, int sizeOfList, int offset){
  int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  int idx = blockDim.x * blockId + threadIdx.x;

  if(idx < numSplitters){
    for(; idx < numSplitters; idx += offset){
      splitters[idx] = list[splittersIndex[idx] % sizeOfList];
    }
  }
  *(splitters -1) = 0;
  *(splitters + numSplitters) = 0xFFFFFFFF;
}

__global__ void copyTheSplitters(float* list, float *splitters, uint *splittersIndex, int numSplitters, int sizeOfList, int offset){
  int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  int idx = blockDim.x * blockId + threadIdx.x;

  if(idx < numSplitters){
    for(; idx < numSplitters; idx += offset){
      splitters[idx] = list[splittersIndex[idx] % sizeOfList];
    }
  }
  //set to -inf, and +inf respectively
  *(splitters- 1) = -1 * CUDART_INF_F;
  *(splitters + numSplitters) = CUDART_INF_F;//INFINITY
}

__global__ void copyTheSplitters(double* list, double *splitters, uint *splittersIndex, int numSplitters, int sizeOfList, int offset){
  int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  int idx = blockDim.x * blockId + threadIdx.x;

  if(idx < numSplitters){
    for(; idx < numSplitters; idx += offset){
      splitters[idx] = list[splittersIndex[idx] % sizeOfList];
    }
  }
  //set to -inf, and +inf respectively
  *(splitters-1) = -1 * CUDART_INF; //-INFINITY
  *(splitters + numSplitters) = CUDART_INF; //INFINITY
}


template<typename T>
void generateSplitters(T* list, T *splitters, int numSplitters, int sizeOfList, int offset, dim3 gridDim,int threadsPerBlock){
  uint *splittersIndex;
 
  cudaMalloc(&splittersIndex, sizeof(uint) * numSplitters);
  generateIndices(splittersIndex, numSplitters, sizeOfList);
  copyTheSplitters<<<gridDim,threadsPerBlock >>>(list, splitters,splittersIndex, numSplitters, sizeOfList, offset);
  cudaFree(splittersIndex);
}
 
