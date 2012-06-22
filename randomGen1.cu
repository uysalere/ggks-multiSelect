#include <curand.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#define FLOAT 1
#define DOUBLE 1
#define INT 1

///**** MonteCarlo.cu ****
// we could vary M sdf& N to find the perf sweet spot

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


/*
struct RandomNumberFunctor :
  public thrust::binary_function<unsigned int, int, double>
{
  unsigned int mainSeed;
  unsigned int sizeOfVector;

  RandomNumberFunctor(unsigned int _mainSeed, int _sizeOfVector) :
    mainSeed(_mainSeed), sizeOfVector(_sizeOfVector){}
  
  __host__ __device__
  float operator()(unsigned int threadIdx)
  {
    unsigned int seed = hash(threadIdx) * mainSeed;

    thrust::default_random_engine rng(seed);
    rng.discard(threadIdx);
    thrust::uniform_real_distribution<double> u(0, sizeOfVector);

    return u(rng);
  }
};
*/

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


///**** MonteCarlo.cu ****
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
void createRandomVectorCurand(T * d_A, int size) {
  curandGenerator_t gen;
  timeval t1;
  uint seed;

  gettimeofday(&t1, NULL);
  seed = t1.tv_usec * t1.tv_sec;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  curandDestroyGenerator(gen);
}

template <typename T>
__global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
  *(in + blockIdx.x*blockDim.x + threadIdx.x) = *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
}


__global__ void enlargeIndexAndGetElements (float * in, uint * out, uint * list, int size) {
  *(out + blockIdx.x * blockDim.x + threadIdx.x) = (uint) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
}


template <typename T>
void printStuff (T * d_list, int size) {
  T * p = (T *) malloc (sizeof (T) * size);

  cudaMemcpy (p, d_list, sizeof(T)* size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 20; i++)
    // printf("%lf\n", *(p+i));
    std::cout << *(p+i) << "\n";
  
  free(p);
}
/*

  template <typename T>
  void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, T min, T max) {

  int maxThreads = 1024;

  double * d_randoms;
  cudaMalloc ((void **) &d_randoms, sizeof (double) * sizeOfSample);

  int numSmallBuckets = (sizeOfSample / (numPivots - 1));
  
  createRandomVectorThrust(d_randoms, sizeOfSample, sizeOfVector);

  // converts randoms floats into elements from necessary indices
  getElements<<<(sizeOfSample/maxThreads), maxThreads>>>(d_randoms, d_list, sizeOfVector);

  pivots[0] = (T) min; 
  pivots[numPivots-1] = (T) max;

  thrust::device_ptr<double>randoms_ptr(d_randoms);
  thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

  cudaThreadSynchronize();

  double holder;

  for (int i = 1; i < numPivots - 1; i++) {    
  cudaMemcpy (&holder, (d_randoms + numSmallBuckets * i), sizeof (double), cudaMemcpyDeviceToHost);     
  *(pivots + i) = (T) holder;
  slopes[i-1] = numSmallBuckets /(pivots[i] - pivots[i-1]);
  }
    
  slopes[numPivots-2] = numSmallBuckets / (pivots[numPivots-1] - pivots[numPivots-2]);

  cudaFree(d_randoms);
  }
*/
template <typename T>
void generatePivots (uint * pivots, double * slopes, uint * d_list, int sizeOfVector, int numPivots, int sizeOfSample, uint min, uint max) {
  
  int maxThreads = 1024;
  float * d_randomFloats;
  uint * d_randomInts;
  
  int pivotOffset = (sizeOfSample / (numPivots - 1));

  cudaMalloc ((void **) &d_randomFloats, sizeof (float) * sizeOfSample);
  
  d_randomInts = (uint *) d_randomFloats;

  createRandomVector (d_randomFloats, sizeOfSample);

  // converts randoms floats into elements from necessary indices
  enlargeIndexAndGetElements<<<(sizeOfSample/maxThreads), maxThreads>>>(d_randomFloats, d_randomInts, d_list, sizeOfVector);


  pivots[0] = min;
  pivots[numPivots-1] = max;

  thrust::device_ptr<T>randoms_ptr(d_randomInts);
  thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

  cudaThreadSynchronize();

  for (int i = 1; i < numPivots - 1; i++) {
    cudaMemcpy (pivots + i, d_randomInts + pivotOffset * i, sizeof (uint), cudaMemcpyDeviceToHost);
    slopes[i-1] = pivotOffset /(pivots[i] - pivots[i-1]);
  }
    
  slopes[numPivots-2] = pivotOffset / (pivots[numPivots-1] - pivots[numPivots-2]);
  
  cudaFree(d_randomInts);
}

template <typename T>
void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, T min, T max) {

  int maxThreads = 1024;
  T * d_randoms;
  int pivotOffset = (sizeOfSample / (numPivots - 1));

  cudaMalloc ((void **) &d_randoms, sizeof (T) * sizeOfSample);
  
  createRandomVector (d_randoms, sizeOfSample);

  // converts randoms floats into elements from necessary indices
  enlargeIndexAndGetElements<<<(sizeOfSample/maxThreads), maxThreads>>>(d_randoms, d_list, sizeOfVector);

  pivots[0] = min;
  pivots[numPivots-1] = max;

  thrust::device_ptr<T>randoms_ptr(d_randoms);
  thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

  cudaThreadSynchronize();

  // TODO:  check to make sure pivots aren't repeated, since that crashes our program currently
  //  note:  need to implement everything in both generatePivots functions

  // set pivots next to the min and max pivots
  cudaMemcpy (pivots + 1, d_randoms + 1, sizeof (T), cudaMemcpyDeviceToHost);
  cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - 2, sizeof (T), cudaMemcpyDeviceToHost);
  slopes[0] = pivotOffset / (double) (pivots[1] - pivots[0]);

  for (int i = 2; i < numPivots - 2; i++) {
    cudaMemcpy (pivots + i, d_randoms + pivotOffset * i, sizeof (T), cudaMemcpyDeviceToHost);

  }




  for (int i = 1; i < numPivots - 1; i++) {

    slopes[i-1] = pivotOffset / (pivots[i] - pivots[i-1]);
  }

  slopes[numPivots-3] = pivotOffset / (double) (pivots[numPivots-2] - pivots[numPivots-3]);
  slopes[numPivots-2] = pivotOffset / (pivots[numPivots-1] - pivots[numPivots-2]);
  
  cudaFree(d_randoms);
}



/***** GENERALIZED VERSION *****/

/*
template <typename T>
__global__ void getElements (double * in, T * list, int size) {
  // *(in + blockIdx.x*numThreads + threadIdx.x) = *(list + ((int) (*(in + blockIdx.x*numThreads + threadIdx.x) * size)));
  // printf("%lf\n", *(in + blockIdx.x*blockDim.x + threadIdx.x));

  *(in + blockIdx.x*blockDim.x + threadIdx.x) = (double) *(list + (int) *(in + blockIdx.x*blockDim.x + threadIdx.x));
  // printf("%lf\n", *(in + blockIdx.x*blockDim.x + threadIdx.x));
}

template <typename T>
void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector, int numPivots, int sizeOfSample, T min, T max) {

  int maxThreads = 1024;

  double * d_randoms;
  cudaMalloc ((void **) &d_randoms, sizeof (double) * sizeOfSample);

  int numSmallBuckets = (sizeOfSample / (numPivots - 1));
  
  createRandomVectorThrust(d_randoms, sizeOfSample, sizeOfVector);

  // converts randoms floats into elements from necessary indices
  getElements<<<(sizeOfSample/maxThreads), maxThreads>>>(d_randoms, d_list, sizeOfVector);

  pivots[0] = (T) min;
  pivots[numPivots-1] = (T) max;

  thrust::device_ptr<double>randoms_ptr(d_randoms);
  thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

  cudaThreadSynchronize();

  double holder;

  for (int i = 1; i < numPivots - 1; i++) {
    cudaMemcpy (&holder, (d_randoms + numSmallBuckets * i), sizeof (double), cudaMemcpyDeviceToHost);
    *(pivots + i) = (T) holder;
    slopes[i-1] = numSmallBuckets /(pivots[i] - pivots[i-1]);
  }
    
  slopes[numPivots-2] = numSmallBuckets / (pivots[numPivots-1] - pivots[numPivots-2]);

  cudaFree(d_randoms);
}
*/


int main() {

  for (int i = 0; i < 1; i++) {
    int sizeOfVector = 100000000;
    int sizeOfSample = 1024;
    int numPivots = 17;
    
    //********* TEST FLOAT **********//
    if (FLOAT) {
      float floatPivots[numPivots];
      double floatSlopes[numPivots - 1];
  
      float * floatList = (float *) malloc (sizeOfVector * sizeof (float));

      // initialize array
      for (int j = 0; j < sizeOfVector; j++)
        floatList[j] =  (float) j;

      float * d_floatList;
      cudaMalloc ((void **) &d_floatList, sizeOfVector * sizeof (float));
      cudaMemcpy(d_floatList, floatList, sizeOfVector * sizeof (float), cudaMemcpyHostToDevice);

      cudaEvent_t start3, stop3;
      float time3;
      cudaEventCreate(&start3);
      cudaEventCreate(&stop3);
      cudaEventRecord(start3,0);

      generatePivots<float>(floatPivots, floatSlopes, d_floatList, sizeOfVector, numPivots, sizeOfSample, floatList[0], floatList[sizeOfVector-1]);

      cudaThreadSynchronize();
      cudaEventRecord(stop3,0);
      cudaEventSynchronize(stop3);
      cudaEventElapsedTime(&time3, start3, stop3);
      cudaEventDestroy(start3);
      cudaEventDestroy(stop3);
    
      free(floatList);
      cudaFree(d_floatList);

      printf("\n\nfloat pivots:\n");
      for (int i = 0; i < numPivots; i++) 
        //std::cout << floatPivots[i] << '\n';
        printf("%f\n", floatPivots[i]);

      printf("\nfloat time = %f\n\n", time3);
    }

    //********* TEST DOUBLE **********//
    if (DOUBLE) {
      double doublePivots[numPivots];
      double doubleSlopes[numPivots - 1];

      double * doubleList = (double *) malloc (sizeOfVector * sizeof (double));

      // initialize array
      for (int j = 0; j < sizeOfVector; j++)
        doubleList[j] =  (double) j;

      double * d_doubleList;
      cudaMalloc ((void **) &d_doubleList, sizeOfVector * sizeof (double));
      cudaMemcpy(d_doubleList, doubleList, sizeOfVector * sizeof (double), cudaMemcpyHostToDevice);

      cudaEvent_t start1, stop1;
      float time1;
      cudaEventCreate(&start1);
      cudaEventCreate(&stop1);
      cudaEventRecord(start1,0);

      generatePivots<double>(doublePivots, doubleSlopes, d_doubleList, sizeOfVector, numPivots, sizeOfSample, doubleList[0], doubleList[sizeOfVector-1]);

      cudaThreadSynchronize();
      cudaEventRecord(stop1,0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&time1, start1, stop1);
      cudaEventDestroy(start1);
      cudaEventDestroy(stop1);
    
      free(doubleList);
      cudaFree(d_doubleList);

      printf("\n\ndouble pivots:\n");
      for (int i = 0; i < numPivots; i++) 
        //std::cout << floatPivots[i] << '\n';
        printf("%lf\n", doublePivots[i]);

      printf("\ndouble time = %f\n\n", time1);

    }

    if (INT) {
      uint intPivots[numPivots];
      double intSlopes[numPivots - 1];

      uint * intList = (uint *) malloc (sizeOfVector * sizeof (uint));

      // initialize array
      for (int j = 0; j < sizeOfVector; j++)
        intList[j] =  (uint) j;

      uint * d_intList;
      cudaMalloc ((void **) &d_intList, sizeOfVector * sizeof (uint));
      cudaMemcpy(d_intList, intList, sizeOfVector * sizeof (uint), cudaMemcpyHostToDevice);

      cudaEvent_t start2, stop2;
      float time2;
      cudaEventCreate(&start2);
      cudaEventCreate(&stop2);
      cudaEventRecord(start2,0);

      generatePivots<uint>(intPivots, intSlopes, d_intList, sizeOfVector, numPivots, sizeOfSample, intList[0], intList[sizeOfVector-1]);

      cudaThreadSynchronize();
      cudaEventRecord(stop2,0);
      cudaEventSynchronize(stop2);
      cudaEventElapsedTime(&time2, start2, stop2);
      cudaEventDestroy(start2);
      cudaEventDestroy(stop2);
    
      free(intList);
      cudaFree(d_intList);

      printf("\n\nint pivots:\n");
      for (int i = 0; i < numPivots; i++) 
        //std::cout << intPivots[i] << '\n';
        printf("%u\n", intPivots[i]);
      
      printf("\nint time = %f\n\n", time2);
      
    }

  }
  return 0;
}
