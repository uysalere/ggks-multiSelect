#include <stdio.h>
#include <curand.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

void createRandomMatrix(float * d_A, int size, int seed) {
  curandGenerator_t gen;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  curandDestroyGenerator(gen);
}

template <typename T>
__global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
  *(in + threadIdx.x) = *(list + ((int) (*(in + threadIdx.x) * size)));
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
    int pivotOffset = (sampleSize / (numPivots - 1));

    cudaMalloc ((void **) &d_randoms, sizeof (T) * sampleSize);
  
    createRandomMatrix (d_randoms, sampleSize, 1);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements<<<1, sampleSize>>>(d_randoms, d_list, numElements);

    pivots[0] = min; 
    pivots[numPivots-1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randoms);
    thrust::sort(randoms_ptr, randoms_ptr + sampleSize);

    cudaThreadSynchronize();

    for (int i = 1; i < numPivots - 1; i++) {
      cudaMemcpy (pivots + i, d_randoms + pivotOffset * i, sizeof (T), cudaMemcpyDeviceToHost);
      slopes[i-1] = pivotOffset /(pivots[i] - pivots[i-1]);
    }
    
    slopes[numPivots-2] = pivotOffset / (pivots[numPivots-1] - pivots[numPivots-2]);
  
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
