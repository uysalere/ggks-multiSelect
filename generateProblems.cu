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

#include <cuda.h>
#include <curand.h>





///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE UINTS
///////////////////////////////////////////////////////////////////
typedef void (*ptrToUintGeneratingFunction)(uint*, uint, curandGenerator_t);

void generateUniformUnsignedIntegers(uint *h_vec, uint numElements, curandGenerator_t generator){
  uint * d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(uint));
  curandGenerate(generator, d_generated,numElements);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}


void generateSortedArrayUints(uint* input, uint length, curandGenerator_t gen)
{
  uint * d_generated;
  cudaMalloc(&d_generated, length * sizeof(uint));
  curandGenerate(gen, d_generated,length);
  thrust::device_ptr<uint>d_ptr(d_generated);
  thrust::sort(d_ptr, d_ptr+length);
  cudaMemcpy(input, d_generated, length * sizeof(uint), cudaMemcpyDeviceToHost);

  cudaFree(d_generated);
}

#define NUMBEROFUINTDISTRIBUTIONS 2
ptrToUintGeneratingFunction arrayOfUintGenerators[NUMBEROFUINTDISTRIBUTIONS] = {&generateUniformUnsignedIntegers,&generateSortedArrayUints};
char* namesOfUintGeneratingFunctions[NUMBEROFUINTDISTRIBUTIONS]={"UNIFORM UNSIGNED INTEGERS","SORTED UINTS"};



///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE FLOATS
///////////////////////////////////////////////////////////////////
typedef void (*ptrToFloatGeneratingFunction)(float*, uint, curandGenerator_t);

void generateUniformFloats(float *h_vec, uint numElements, curandGenerator_t generator){
  float * d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(float));
  curandGenerateUniform(generator, d_generated,numElements);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}

void generateNormalFloats(float* h_vec, uint numElements, curandGenerator_t generator){
  float *d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(float));
  curandGenerateNormal(generator, d_generated, numElements,0,1);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}
__global__ void setAllToValue(float* input, int length, float value, int offset)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < length){
    int i;
    for(i=idx; i<length; i+=offset){
      input[i] = value;
    }
  }
}

__global__ void createVector(float* input, int length, int firstVal, int numFirstVal, int secondVal, int numSecondVal, int offset){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < length){
    int i;
    for(i=idx; i< numFirstVal; i+=offset){
      input[i] = firstVal;
    }
    syncthreads();

    for(i=idx; i< (numFirstVal + numSecondVal); i+=offset){
      if(i >= numFirstVal){
        input[i] = secondVal;
      }
    }
  }
}


__global__ void scatter(float* input, int length, int offset){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if( (idx < length) && ((idx % 2) == 0) ){
    int i, j;
    int lastEvenIndex = (length-1) - ( (length-1) % 2);
    float temp;

    for(i=idx; i< (length/2); i+=offset){
      //if i is even
      if( (i % 2) == 0)
        j = lastEvenIndex - i;

      //switch i and j
      temp = input[i];
      input[i] = input[j];
      input[j] = temp;
    }
  }
}

//takes in a device input vector
void generateOnesTwosNoisyFloats(float* input, int length, int firstVal, int firstPercent, int secondVal, int secondPercent)
{
  float* devVec;
  cudaMalloc(&devVec, sizeof(float) * length);

  int numFirstVal = (length * firstPercent) / 100;
  int numSecondVal = length - numFirstVal;

  //get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;
  int maxThreads = deviceProp.maxThreadsPerBlock;
  int offset = blocks * maxThreads;

  //create vector of ones and twos
  createVector<<<blocks, maxThreads>>>(devVec, length, firstVal, numFirstVal, secondVal, numSecondVal, offset);

  //shuffle the elements of the vector
  scatter<<<blocks, maxThreads>>>(devVec, length, offset);

  cudaMemcpy(input, devVec, sizeof(float)*length, cudaMemcpyDeviceToHost);
  cudaFree(devVec);
} 

void generateOnesTwosFloats(float* input, uint length, curandGenerator_t gen) 
{
  float* devVec;
  cudaMalloc(&devVec, sizeof(float) * length);

  int numFirstVal = (length * 95) / 100;
  int numSecondVal = length - numFirstVal;

  //get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;
  int maxThreads = deviceProp.maxThreadsPerBlock;
  int offset = blocks * maxThreads;

  //create vector of ones and twos
  createVector<<<blocks, maxThreads>>>(devVec, length, 2, numFirstVal, 1, numSecondVal, offset);

  //shuffle the elements of the vector
  scatter<<<blocks, maxThreads>>>(devVec, length, offset);

  cudaMemcpy(input, devVec, sizeof(float)*length, cudaMemcpyDeviceToHost);
  cudaFree(devVec);
}

//sets everything in this vector to 1.0
void generateAllOnesFloats(float* input, uint length, curandGenerator_t gen)
{
  float* devVec;
  cudaMalloc(&devVec, sizeof(float) * length);

  //get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;
  int maxThreads = deviceProp.maxThreadsPerBlock;
  int offset = blocks * maxThreads;

  setAllToValue<<<blocks, maxThreads>>>(devVec, length, 1.0, offset);

  cudaMemcpy(input, devVec, sizeof(float)*length, cudaMemcpyDeviceToHost);

  cudaFree(devVec);
}

//add first and second and store it in first
__global__ void vectorAdd(float* first, float* second, int length, int offset){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < length){
    int i;
    
    for(i=idx; i<length; i+=offset){
      first[i] = first[i] + second[i];
    }
  }
}

void generateNoisyVector(float* input, uint length, curandGenerator_t gen){
  float* devInput;
  cudaMalloc(&devInput, sizeof(float)*length);

 
  curandGenerateNormal(gen, devInput, length, 0.0, 0.01);

  //get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;
  int maxThreads = deviceProp.maxThreadsPerBlock;
  int offset = blocks * maxThreads;

  float* hostVec = (float*)malloc(sizeof(float)*length);
  float* devVec;
  cudaMalloc(&devVec, sizeof(float) *length);

  generateOnesTwosNoisyFloats(hostVec, length, 1.0, 20, 0.0, 80);
  cudaMemcpy(devVec, hostVec, sizeof(float)*length, cudaMemcpyHostToDevice);

  vectorAdd<<<blocks, maxThreads>>>(devInput, devVec, length, offset);

  cudaMemcpy(input, devInput, sizeof(float)*length, cudaMemcpyDeviceToHost);

  cudaFree(devInput);
  cudaFree(devVec);
  free(hostVec);
}

struct multiplyByMillion
{
  __host__ __device__
  void operator()(float &key){
    key = key * 1000000;
  }
};

void generateHugeUniformFloats(float* input, uint length, curandGenerator_t gen){
  float* devInput;
  cudaMalloc(&devInput, sizeof(float) * length);

  
  curandGenerateUniform(gen, devInput, length);
  thrust::device_ptr<float> dev_ptr(devInput);
  thrust::for_each( dev_ptr, dev_ptr + length, multiplyByMillion());
  cudaMemcpy(input, devInput, sizeof(float)*length, cudaMemcpyDeviceToHost);

  cudaFree(devInput);
}


void generateNormalFloats100(float* input, uint length, curandGenerator_t gen){
  float* devInput;
  cudaMalloc(&devInput, sizeof(float) * length);
  curandGenerateNormal(gen, devInput, length, 0.0, 100.0);
  cudaMemcpy(input, devInput, sizeof(float)*length, cudaMemcpyDeviceToHost);
  cudaFree(devInput);
}


__global__ void createAbsolute(float* input, int length, int offset){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < length){
    int i;
    for(i=idx; i< length; i+=offset){
      input[i] = abs(input[i]);
    }
  }
}


void generateHalfNormalFloats(float* input, uint length, curandGenerator_t gen){
  float* devInput;
  cudaMalloc(&devInput, sizeof(float) * length);

  
  curandGenerateNormal(gen, devInput, length, 0.0, 1.0);

  //get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;
  int maxThreads = deviceProp.maxThreadsPerBlock;
  int offset = blocks * maxThreads;

  createAbsolute<<<blocks, maxThreads>>>(devInput, length, offset);

  cudaMemcpy(input, devInput, sizeof(float)*length, cudaMemcpyDeviceToHost);

  cudaFree(devInput);
}


struct makeSmallFloat
{
  __host__ __device__
  void operator()(uint &key){
    key = key & 0x80EFFFFF;
  }
};


void generateBucketKillerFloats(float *h_vec, uint numElements, curandGenerator_t generator){
  int i;
  float * d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(float));
  curandGenerateUniform(generator, d_generated,numElements);
  thrust::device_ptr<unsigned int> dev_ptr((uint *)d_generated);
  thrust::for_each( dev_ptr, dev_ptr + numElements, makeSmallFloat());
  thrust::sort(dev_ptr,dev_ptr + numElements);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(float), cudaMemcpyDeviceToHost);
 
  for(i = -126; i < 127; i++){
    h_vec[i + 126] = pow(2.0,(float)i);
  }
  cudaFree(d_generated);
}

#define NUMBEROFFLOATDISTRIBUTIONS 9
ptrToFloatGeneratingFunction arrayOfFloatGenerators[NUMBEROFFLOATDISTRIBUTIONS] = {&generateUniformFloats, &generateNormalFloats,&generateBucketKillerFloats,
                                                                                   &generateHalfNormalFloats,&generateNormalFloats100, &generateHugeUniformFloats,
                                                                                   &generateNoisyVector,&generateAllOnesFloats,&generateOnesTwosFloats};

char* namesOfFloatGeneratingFunctions[NUMBEROFFLOATDISTRIBUTIONS]={"UNIFORM FLOATS","NORMAL FLOATS","KILLER FLOATS",
                                                                   "HALF NORMAL FLOATS","NORMAL FLOATS 100", "HUGE UNIFORM FLOATS",
                                                                   "NOISY FLOATS","ALL ONES FLOATS","ONES TWOS FLOAT"};


///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE DOUBLES
///////////////////////////////////////////////////////////////////

typedef void (*ptrToDoubleGeneratingFunction)(double*, uint, curandGenerator_t);

void generateUniformDoubles(double *h_vec, uint numElements, curandGenerator_t generator){
  double * d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(double));
  curandGenerateUniformDouble(generator, d_generated,numElements);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}

void generateNormalDoubles(double* h_vec, uint numElements, curandGenerator_t gen){
  double *d_generated;
  cudaMalloc(&d_generated, numElements * sizeof(double));
  curandGenerateNormalDouble(gen, d_generated, numElements,0,1);
  cudaMemcpy(h_vec, d_generated, numElements * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_generated);
}

struct makeSmallDouble
{
  __host__ __device__
  void operator()(unsigned long long &key){
    key = key & 0x800FFFFFFFFFFFFF;
  }
};

void generateBucketKillerDoubles(double *h_vec, uint numElements, curandGenerator_t generator){
   int i;
   double * d_generated;
   cudaMalloc(&d_generated, numElements * sizeof(double));
   curandGenerateUniformDouble(generator, d_generated,numElements);
   thrust::device_ptr<unsigned long long> dev_ptr((unsigned long long *)d_generated);
   thrust::for_each( dev_ptr, dev_ptr + numElements, makeSmallDouble());
    thrust::sort(dev_ptr,dev_ptr + numElements);
   cudaMemcpy(h_vec, d_generated, numElements * sizeof(double), cudaMemcpyDeviceToHost);
 
   for(i = -1022; i < 1023; i++){
     h_vec[i + 1022] = pow(2.0,(double)i);
   }
   cudaFree(d_generated);
 }

#define NUMBEROFDOUBLEDISTRIBUTIONS 3
ptrToDoubleGeneratingFunction arrayOfDoubleGenerators[NUMBEROFDOUBLEDISTRIBUTIONS] = {&generateUniformDoubles,&generateNormalDoubles,
                                                                                      &generateBucketKillerDoubles};
char* namesOfDoubleGeneratingFunctions[NUMBEROFDOUBLEDISTRIBUTIONS]={"UNIFORM DOUBLES","NORMAL DOUBLES", "KILLER DOUBLES"};



template<typename T> void* returnGenFunctions(){
  if(typeid(T) == typeid(uint)){
    return arrayOfUintGenerators;
  }
  else if(typeid(T) == typeid(float)){
    return arrayOfFloatGenerators;
  }
  else{
    return arrayOfDoubleGenerators;
  }
}


template<typename T> char** returnNamesOfGenerators(){
  if(typeid(T) == typeid(uint)){
    return &namesOfUintGeneratingFunctions[0];
  }
  else if(typeid(T) == typeid(float)){
    return &namesOfFloatGeneratingFunctions[0];
  }
  else {
    return &namesOfDoubleGeneratingFunctions[0];
  }
}


void printDistributionOptions(uint type){
  switch(type){
  case 1:
    PrintFunctions::printArray(returnNamesOfGenerators<float>(), NUMBEROFFLOATDISTRIBUTIONS);
    break;
  case 2:
    PrintFunctions::printArray(returnNamesOfGenerators<double>(), NUMBEROFDOUBLEDISTRIBUTIONS);
    break;
  case 3:
    PrintFunctions::printArray(returnNamesOfGenerators<uint>(), NUMBEROFUINTDISTRIBUTIONS);
    break;
  default:
    printf("You entered and invalid option\n");
    break;
  }
}
