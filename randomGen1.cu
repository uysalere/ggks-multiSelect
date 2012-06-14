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

__global__ void enlargeIndexAndGetElements (float * in, float * list, int size) {
  *(in + threadIdx.x) = *(list + ((int) (*(in + threadIdx.x) * size)));

  // printf("%f\n", *(in + threadIdx.x));
}

void printStuff (float * d_list, int size) {
  float * p = (float *) malloc (sizeof (float) * size);

  cudaMemcpy (p, d_list, sizeof (float) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 20; i++)
    printf("%f\n", *(p+i));
  
  free(p);
}

void generateSplitters (float * splitters, float * slopes, float * d_list, int numElements, int numSplitters, int sampleSize) {

  float * d_randoms;
  
  cudaMalloc ((void **) &d_randoms, sizeof (float) * sampleSize);
  
  createRandomMatrix (d_randoms, sampleSize, 1);

  // turn randoms floats into necessary indices
  enlargeIndexAndGetElements<<<1, sampleSize>>>(d_randoms, d_list, numElements);

  thrust::device_ptr<float>list_ptr(d_list);
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float> > extrema = thrust::minmax_element (list_ptr, list_ptr + numElements);

  splitters[0] = *(extrema.first); 
  splitters[numSplitters-1] = *(extrema.second);

  thrust::device_ptr<float>randoms_ptr(d_randoms);
  thrust::sort(randoms_ptr, randoms_ptr + sampleSize);

  //cudaThreadSynchronize();
  //printStuff (d_randoms, sampleSize);

  for (int i = 1; i < numSplitters - 1; i++) {
    // splitters[i] = (sampleSize / (numSplitters - 1) * i)
    cudaMemcpy (splitters + i, d_randoms +(sampleSize / (numSplitters - 1) * i), sizeof (float), cudaMemcpyDeviceToHost);
    slopes[i-1] = (splitters[i] - splitters[i-1]) / (sampleSize / (numSplitters - 1));
  }
  slopes[numSplitters-2] = (splitters[numSplitters-1] - splitters[numSplitters-2]) / (sampleSize / (numSplitters - 1));
  
  
  /*
  for (int i = 0; i < sampleSize; i++) {
    cudaMemcpy (randoms + i, d_list + randomIndices[i], sizeof (float), cudaMemcpyDeviceToHost); 
    printf("%f\n", randoms[i]);
  }

  thrust::sort (randoms, randoms + sampleSize);

    for (int i = 0; i < 100; i++)
    {
      
    }
  
  /*
  thrust::device_ptr<int>dev_ptr2(d_randoms);
  thrust::sort (dev_ptr2, dev_ptr2 + sampleSize);
  
  for (int i = 0; i < 100; i++)
    {
      //int x = *(dev_ptr + i);
      int x = *(d_randoms+i);
      printf("%d\n", x);
    }
  */
  
}



int main() {
  int numElements = 10000000;
  int sampleSize = 1024;
  int numSplitters = 9;
  
  float splitters[numSplitters];
  float slopes[numSplitters - 1];
  
  float * list = (float *) malloc (numElements * sizeof (float));

  // initialize array
  for (int i = 0; i < numElements; i++)
    list[i] = (float) i;

  float * d_list;
  cudaMalloc ((void **) &d_list, numElements * sizeof (float));
  cudaMemcpy(d_list, list, numElements * sizeof (float), cudaMemcpyHostToDevice);
  
  generateSplitters (splitters, slopes, d_list, numElements, numSplitters, sampleSize);
  //  generateSlopes (slopes, splitters, numSplitters);

  for (int i = 0; i < numSplitters; i++)
    printf ("splitter[%d] = %f\n", i, splitters[i]);

  for (int i = 0; i < numSplitters-1; i++)
    printf ("slopes[%d] = %f\n", i, slopes[i]);
    
  return 0;
}
