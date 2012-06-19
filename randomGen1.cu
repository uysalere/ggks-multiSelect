#include <curand.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#define TYPE int

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
///**** MonteCarlo.cu ****

void createRandomVectorThrust(double * d_vec, int sizeOfSample, int sizeOfVector) {
  timeval t1;
  uint seed;

  gettimeofday(&t1, NULL);
  seed = t1.tv_usec * t1.tv_sec;
  
  thrust::device_ptr<double> d_ptr(d_vec);
  thrust::transform(thrust::counting_iterator<uint>(0),thrust::counting_iterator<uint>(sizeOfSample),
                    d_ptr, RandomNumberFunctor(seed, sizeOfVector));
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
__global__ void getElements (double * in, T * list, int size) {
  // *(in + blockIdx.x*numThreads + threadIdx.x) = *(list + ((int) (*(in + blockIdx.x*numThreads + threadIdx.x) * size)));
  // printf("%lf\n", *(in + blockIdx.x*blockDim.x + threadIdx.x));

 *(in + blockIdx.x*blockDim.x + threadIdx.x) = (double) *(list + (int) *(in + blockIdx.x*blockDim.x + threadIdx.x));
 //  printf("%lf\n", *(in + blockIdx.x*blockDim.x + threadIdx.x));
}

void printStuff (double * d_list, int size) {
  double * p = (double *) malloc (sizeof (double) * size);

  cudaMemcpy (p, d_list, sizeof(double)* size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 20; i++)
    // printf("%lf\n", *(p+i));
    std::cout << *(p+i) << "\n";
  
  free(p);
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



int main() {

  for (int i = 0; i < 1; i++) {
    int sizeOfVector = 1000000;
    int sizeOfSample = 1024;
    int numSplitters = 9;
  
    TYPE splitters[numSplitters];
    double slopes[numSplitters - 1];
  
    TYPE * list = (TYPE *) malloc (sizeOfVector * sizeof (TYPE));

    // initialize array
    for (int j = 0; j < sizeOfVector; j++)
      list[j] =  (TYPE) j;

    TYPE * d_list;
    cudaMalloc ((void **) &d_list, sizeOfVector * sizeof (TYPE));
    cudaMemcpy(d_list, list, sizeOfVector * sizeof (TYPE), cudaMemcpyHostToDevice);

    cudaEvent_t start3, stop3;
    float time3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3,0);

    generatePivots<TYPE>(splitters, slopes, d_list, sizeOfVector, numSplitters, sizeOfSample, list[0], list[sizeOfVector-1]);

    cudaThreadSynchronize();
    cudaEventRecord(stop3,0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&time3, start3, stop3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    
    free(list);
    cudaFree(d_list);
   
    for (int i = 0; i < numSplitters; i++) 
       std::cout << splitters[i] << '\n';
      // printf("%lf\n", splitters[i]);

    printf("time = %f\n", time3);
  

  }
  return 0;
}
