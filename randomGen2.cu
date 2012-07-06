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
      while (slopes[stop] == 0 || isinf(slopes[stop]) && stop < numPivots) 
        stop++;
      // now slopes[i] is the first repeated zero or inf in a row, and
      //  slopes[stop-1] is the last zero or inf slope
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
        floatList[j] =  (float) (j % 4);

      float * d_floatList;
      cudaMalloc ((void **) &d_floatList, sizeOfVector * sizeof (float));
      cudaMemcpy(d_floatList, floatList, sizeOfVector * sizeof (float), cudaMemcpyHostToDevice);

      thrust::device_ptr<float>float_ptr(d_floatList);
      thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float> > result = thrust::minmax_element(float_ptr, float_ptr + sizeOfVector);
      float floatMin = *result.first;
      float floatMax = *result.second;

      cudaEvent_t start3, stop3;
      float time3;
      cudaEventCreate(&start3);
      cudaEventCreate(&stop3);
      cudaEventRecord(start3,0);

      generatePivots<float>(floatPivots, floatSlopes, d_floatList, sizeOfVector, numPivots, sizeOfSample, 1024, floatMin, floatMax);

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

      thrust::device_ptr<double>double_ptr(d_doubleList);
      thrust::pair<thrust::device_ptr<double>, thrust::device_ptr<double> > result = thrust::minmax_element(double_ptr, double_ptr + sizeOfVector);
      float doubleMin = *result.first;
      float doubleMax = *result.second;

      cudaEvent_t start1, stop1;
      float time1;
      cudaEventCreate(&start1);
      cudaEventCreate(&stop1);
      cudaEventRecord(start1,0);

      generatePivots<double>(doublePivots, doubleSlopes, d_doubleList, sizeOfVector, numPivots, sizeOfSample, 1024, doubleMin, doubleMax);

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
      for (uint j = 0; j < sizeOfVector; j++)
        intList[j] = j%4; 

      uint * d_intList;
      cudaMalloc ((void **) &d_intList, sizeOfVector * sizeof (uint));
      cudaMemcpy(d_intList, intList, sizeOfVector * sizeof (uint), cudaMemcpyHostToDevice);

      thrust::device_ptr<uint>uint_ptr(d_intList);
      thrust::pair<thrust::device_ptr<uint>, thrust::device_ptr<uint> > result = thrust::minmax_element(uint_ptr, uint_ptr + sizeOfVector);
      float uintMin = *result.first;
      float uintMax = *result.second;

      cudaEvent_t start2, stop2;
      float time2;
      cudaEventCreate(&start2);
      cudaEventCreate(&stop2);
      cudaEventRecord(start2,0);

      generatePivots<uint>(intPivots, intSlopes, d_intList, sizeOfVector, numPivots, sizeOfSample, 1024, uintMin, uintMax);

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

      printf("\nint slopes:\n");
      for (int i = 0; i < numPivots - 1; i++)
        printf("%lf\n", intSlopes[i]);

      fixSlopes(intSlopes, numPivots);

      printf("\nint slopes:\n");
      for (int i = 0; i < numPivots - 1; i++)
        printf("%lf\n", intSlopes[i]);

      printf("\nint time = %f\n\n", time2);
      
    }

  }
  return 0;
}
