#include <curand.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

///**** MonteCarlo.cu ****
// we could vary M & N to find the perf sweet spot

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


///**** MonteCarlo.cu ****

template <typename T>
void createRandomVectorThrust(T * d_vec, int size) {
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
__global__ void enlargeIndexAndGetElements (T * in, T * list, int size, int numThreads) {
  //*(in + blockIdx.x*numThreads + threadIdx.x) = *(list + blockIdx.x*numThreads + threadIdx.x);
  *(in + blockIdx.x*numThreads + threadIdx.x) = *(list + ((int) (*(in + blockIdx.x*numThreads + threadIdx.x) * size)));
}

void printStuff (float * d_list, int size) {
  float * p = (float *) malloc (sizeof (float) * size);

  cudaMemcpy (p, d_list, sizeof (float) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 20; i++)
    printf("%f\n", *(p+i));
  
  free(p);
}

  template <typename T>
  void generatePivots (T * pivots, double * slopes, T * d_list, int numElements, int numPivots, int sampleSize, T min, T max) {

    T * d_randoms;
    int numThreads = 1024;
    int numSmallBuckets = (sampleSize / (numPivots - 1));
    cudaMalloc ((void **) &d_randoms, sizeof (T) * sampleSize);
  
    //createRandomVectorCurand (d_randoms, sampleSize);
    createRandomVectorThrust (d_randoms, sampleSize);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<(sampleSize/numThreads), numThreads>>>(d_randoms, d_list, numElements, numThreads);

    pivots[0] = min; 
    pivots[numPivots-1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randoms);
    thrust::sort(randoms_ptr, randoms_ptr + sampleSize);

    cudaThreadSynchronize();

    printStuff(d_randoms, 1024);

    for (int i = 1; i < numPivots - 1; i++) {
      cudaMemcpy (pivots + i, d_randoms + numSmallBuckets * i, sizeof (T), cudaMemcpyDeviceToHost);
      slopes[i-1] = numSmallBuckets /(pivots[i] - pivots[i-1]);
    }
    
    slopes[numPivots-2] = numSmallBuckets / (pivots[numPivots-1] - pivots[numPivots-2]);
  
    cudaFree(d_randoms);
  }



int main() {

  for (int i = 0; i < 1; i++) {
    int numElements = 1000000;
    int sampleSize = 1024;
    int numSplitters = 9;
  
    float splitters[numSplitters];
    double slopes[numSplitters - 1];
  
    float * list = (float *) malloc (numElements * sizeof (float));

    // initialize array
    for (int j = 0; j < numElements; j++)
      list[j] = (float) j;

    float * d_list;
    cudaMalloc ((void **) &d_list, numElements * sizeof (float));
    cudaMemcpy(d_list, list, numElements * sizeof (float), cudaMemcpyHostToDevice);

    cudaEvent_t start3, stop3;
    float time3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3,0);

    /*    template <typename T>
          void generatePivots (T * pivots, double * slopes, T * d_list, int numElements, int numPivots, int sampleSize, T min, T max) */
       
    generatePivots<float>(splitters, slopes, d_list, numElements, numSplitters, sampleSize, list[0], list[numElements-1]);

    cudaThreadSynchronize();
    cudaEventRecord(stop3,0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&time3, start3, stop3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    
    free(list);
    cudaFree(d_list);
   
    printf("time = %f\n", time3);
  }

  
  return 0;
}
