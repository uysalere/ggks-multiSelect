#include <stdio.h>
#include <curand.h>
#include <thrust/extrema.h>

void createRandomMatrix(float * A, int size, int seed) {
  float *d_A;
  // float *h_A = (float *) malloc (size * sizeof(float));
  curandGenerator_t gen;
  size_t size_d_A = size * sizeof(float);

  cudaMalloc((void **) &d_A, size_d_A);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  cudaMemcpy(A, d_A, size_d_A, cudaMemcpyDeviceToHost);

  curandDestroyGenerator(gen);
  cudaFree(d_A);
  //  free(h_A);
}

void generateSplitters (float * splitters, float * d_list, int numElements, int numSplitters, int sampleSize, int randomSize) {

  int randomElements[randomSize];
  float randoms[randomSize];

  createRandomMatrix (randoms, randomSize, 1);

  for (int i = 0; i < randomSize; i++)
    randomElements[i] = (int) (randoms[i] * (float) numElements);


  float * d_minmax;
  cudaMalloc (&d_minmax, 2 * sizeof (float));

  thrust::device_ptr<float>dev_ptr(d_list);
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float> > extrema = thrust::minmax_element (dev_ptr, dev_ptr + numElements);

  splitters[0] = *(extrema.first);
  splitters[numSplitters-1] = *(extrema.second);
  

  
}



int main() {
  int numElements = 10000000;
  int sampleSize = 1024;
  int numSplitters = 9;
  int numRands = 1024;
  
  float splitters[numSplitters];
  float slopes[numSplitters - 1];
  
  float * list = (float *) malloc (numElements * sizeof (float));

  // initialize array
  for (int i = 0; i < numElements; i++)
    list[i] = (float) i;

  float * d_list;
  cudaMalloc ((void **) &d_list, numElements * sizeof (float));
  cudaMemcpy(d_list, list, numElements * sizeof (float), cudaMemcpyHostToDevice);
  
  generateSplitters (splitters, d_list, numElements, numSplitters, sampleSize, numRands);
  //  generateSlopes (slopes, splitters, numSplitters);

  for (int i = 0; i < numSplitters; i++)
    printf ("splitter[%d] = %f\n", i, splitters[i]);

  
  return 0;
}
