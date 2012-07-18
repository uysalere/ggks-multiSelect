#include <curand.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <math.h>

#define FLOAT 0
#define DOUBLE 0
#define INT 1

#define MAX_THREADS_PER_BLOCK 1024

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

  cudaFree(d_randoms);
}

void fixSlopes (double * slopes, int numPivots) {
  int stop;
  double tempSlope;

  for (int i = 0; i < numPivots - 1; i++) {
    if (slopes[i] == 0 || isinf(slopes[i])) {
      stop = i+1;
      while (slopes[stop+1] == 0 || isinf(slopes[stop+1]) && stop+1 < numPivots) 
        stop++;

      // now slopes[i] is the first repeated zero or inf of the group, and
      //  slopes[stop] is the last zero or inf slope
      if (stop == numPivots - 1) {
        printf("here\n");
        // steal slope from below
        tempSlope = slopes[i-1] / (stop - i);
      }
      else
        // steal slope from above
        tempSlope = slopes[stop] / (stop - i);

      // put in tempSlope where it was 0 or inf before
      for (int j = i; j < stop; j++) {
        slopes[j] = tempSlope;
      }
    }
  }
}

template <typename T>
float runTest(int numPivots, int sizeOfVector, int sizeOfSample) {
  T pivots[numPivots];
  double slopes[numPivots - 1];

  T * list = (T *) malloc (sizeOfVector * sizeof (T));

  // initialize array
  for (int j = 0; j < sizeOfVector; j++)
    list[j] =  (T) (j % 4);

  T * d_list;
  cudaMalloc (&d_list, sizeOfVector * sizeof (T));
  cudaMemcpy(d_list, list, sizeOfVector * sizeof (T), cudaMemcpyHostToDevice);

  thrust::device_ptr<T>list_ptr(d_list);
  thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result = thrust::minmax_element(list_ptr, list_ptr + sizeOfVector);

  T min = *result.first;
  T max = *result.second;

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  generatePivots<T>(pivots, slopes, d_list, sizeOfVector, numPivots, sizeOfSample, 1024, min, max);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
   
  free(list);
  cudaFree(d_list);

  printf("pivots:\n");
  for (int i = 0; i < numPivots; i++) 
    std::cout << pivots[i] << '\n';

  printf("\nslopes before:\n");
  for (int i = 0; i < numPivots-1; i++) 
    std::cout << slopes[i] << '\n';
   
  fixSlopes(slopes, numPivots);

  printf("\nslopes after:\n");
  for (int i = 0; i < numPivots-1; i++) 
    std::cout << slopes[i] << '\n';

  return time;
}


int main() {

  for (int i = 0; i < 1; i++) {
    int sizeOfVector = 100000000;
    int sizeOfSample = 1024;
    int numPivots = 17;
    
    //********* TEST FLOAT **********//
    if (FLOAT) 
      printf("\nfloat time = %f\n\n", runTest<float>(numPivots, sizeOfVector, sizeOfSample));
    

    //********* TEST DOUBLE **********//
    if (DOUBLE)
      printf("\ndouble time = %f\n\n", runTest<double>(numPivots, sizeOfVector, sizeOfSample));

    //*********** TEST INT ***********//
    if (INT)    
      printf("\nint time = %f\n\n", runTest<int>(numPivots, sizeOfVector, sizeOfSample));
  }

  return 0;
}
