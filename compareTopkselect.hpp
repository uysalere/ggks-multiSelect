#ifndef COMPARETOPKSELECT_H_
#define COMPARETOPKSELECT_H_

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include <algorithm>
//Include various thrust items that are used
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>

//various functions, include the functions
//that print numbers in binary.
#include "printFunctions.cu"

//the algorithms
#include "bucketMultiselect.cu"
#include "randomizedTopkSelect.cu"

#include "generateProblems.cu"
#include "multiselectTimingFunctions.cu"

#define NUMBEROFALGORITHMS 3
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] = 
  {"Sort and Choose Topkselect", "Bucket Topkselect", "Randomized Topkselect"};

namespace CompareTopkselect {

  using namespace std;

  template<typename T>
  void compareMultiselectAlgorithms(uint size, uint numKs, uint numTests
                                    , uint *algorithmsToTest, uint generateType, char* fileNamecsv
                                    , T* data = NULL) {

    // allocate space for operations
    T *h_vec, *h_vec_copy;
    float timeArray[NUMBEROFALGORITHMS][numTests];
    T * resultsArray[NUMBEROFALGORITHMS][numTests];
    float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
    uint winnerArray[numTests];
    uint timesWon[NUMBEROFALGORITHMS];
    uint i,j,m,x;
    int runOrder[NUMBEROFALGORITHMS];

    unsigned long long seed;
    results_t<T> *temp;
    ofstream fileCsv;
    timeval t1;
 
    typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint);
    typedef void (*ptrToGeneratingFunction)(T*, uint, curandGenerator_t);

    //these are the functions that can be called
    ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = 
      {
        &timeSortAndChooseTopkselect<T>,
        &timeBucketTopkselect<T>, 
        &timeRandomizedTopkselect<T>
      };
  
    ptrToGeneratingFunction *arrayOfGenerators;
    char** namesOfGeneratingFunctions;
  
    // this is the array of names of functions that generate problems of this type, 
    // ie float, double, or uint
    namesOfGeneratingFunctions = returnNamesOfGenerators<T>();
    arrayOfGenerators = (ptrToGeneratingFunction *) returnGenFunctions<T>();

    printf("Files will be written to %s\n", fileNamecsv);
    fileCsv.open(fileNamecsv, ios_base::app);
  
    //zero out the totals and times won
    bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
    bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));

    //allocate space for h_vec, and h_vec_copy
    h_vec = (T *) malloc(size * sizeof(T));
    h_vec_copy = (T *) malloc(size * sizeof(T));

    //create the random generator.
    curandGenerator_t generator;
    srand(unsigned(time(NULL)));

    printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);

    /***********************************************/
    /*********** START RUNNING TESTS ************
  /***********************************************/

    for(i = 0; i < numTests; i++) {
      //cudaDeviceReset();
      gettimeofday(&t1, NULL);
      seed = t1.tv_usec * t1.tv_sec;
    
      for(m = 0; m < NUMBEROFALGORITHMS;m++)
        runOrder[m] = m;
    
      std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
      fileCsv << size << "," << numKs << "," << 
        namesOfGeneratingFunctions[generateType] << "," ;

      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(generator,seed);
      printf("Running test %u of %u for size: %u and numK: %u\n", i + 1, 
             numTests, size, numKs);

      //generate the random vector using the specified distribution
      if(data == NULL) 
        arrayOfGenerators[generateType](h_vec, size, generator);
      else
        h_vec = data;

      //copy the vector to h_vec_copy, which will be used to restore it later
      memcpy(h_vec_copy, h_vec, size * sizeof(T));

      winnerArray[i] = 0;
      float currentWinningTime = INFINITY;
      //run the various timing functions
      for(x = 0; x < NUMBEROFALGORITHMS; x++){
        j = runOrder[x];
        if(algorithmsToTest[j]){

          //run timing function j
          printf("TESTING: %u\n", j);
          temp = arrayOfTimingFunctions[j](h_vec_copy, size, numKs);

          //record the time result
          timeArray[j][i] = temp->time;
          //record the value returned
          resultsArray[j][i] = temp->vals;
          //update the current "winner" if necessary
          if(timeArray[j][i] < currentWinningTime){
            currentWinningTime = temp->time;
            winnerArray[i] = j;
          }

          //perform clean up 
          free(temp);
          memcpy(h_vec_copy, h_vec, size * sizeof(T));
        }
      }

      curandDestroyGenerator(generator);
      for(x = 0; x < NUMBEROFALGORITHMS; x++)
        if(algorithmsToTest[x])
          fileCsv << namesOfMultiselectTimingFunctions[x] << "," << timeArray[x][i] << ",";

      // check for errors, and output information to recreate problem
      uint flag = 0;
      for(m = 1; m < NUMBEROFALGORITHMS;m++)
        if(algorithmsToTest[m])
          for (j = 0; j < numKs; j++) {
            if(resultsArray[m][i][j] != resultsArray[0][i][j]) {
              flag++;
              fileCsv << "\nERROR ON TEST " << i << " of " << numTests << " tests!!!!!\n";
              fileCsv << "vector size = " << size << "\nvector seed = " << seed << "\n";
              fileCsv << "numKs = " << numKs << "\n";
              fileCsv << "wrong k = " << j << 
                " wrong result = " << resultsArray[m][i][j] << " correct result = " <<  
                resultsArray[0][i][j] << "\n";
              std::cout <<namesOfMultiselectTimingFunctions[m] <<
                " did not return the correct answer on test " << i + 1 << " at k[" << j << 
                "].  It got "<< resultsArray[m][i][j];
              std::cout << " instead of " << resultsArray[0][i][j] << ".\n" ;
              std::cout << "RESULT:\t";
              PrintFunctions::printBinary(resultsArray[m][i][j]);
              std::cout << "Right:\t";
              PrintFunctions::printBinary(resultsArray[0][i][j]);
            }
          }

      fileCsv << flag << "\n";
    }
  
    //calculate the total time each algorithm took
    for(i = 0; i < numTests; i++)
      for(j = 0; j < NUMBEROFALGORITHMS;j++)
        if(algorithmsToTest[j])
          totalTimesPerAlgorithm[j] += timeArray[j][i];

    //count the number of times each algorithm won. 
    for(i = 0; i < numTests;i++)
      timesWon[winnerArray[i]]++;

    printf("\n\n");

    //print out the average times
    for(i = 0; i < NUMBEROFALGORITHMS; i++)
      if(algorithmsToTest[i])
        printf("%-20s averaged: %f ms\n", namesOfMultiselectTimingFunctions[i], totalTimesPerAlgorithm[i] / numTests);

    for(i = 0; i < NUMBEROFALGORITHMS; i++)
      if(algorithmsToTest[i])
        printf("%s won %u times\n", namesOfMultiselectTimingFunctions[i], timesWon[i]);

    // free results
    for(i = 0; i < numTests; i++) 
      for(m = 0; m < NUMBEROFALGORITHMS; m++) 
        if(algorithmsToTest[m])
          free(resultsArray[m][i]);

    //free h_vec and h_vec_copy
    if(data == NULL) 
      free(h_vec);
    free(h_vec_copy);

    //close the file
    fileCsv.close();
  }


  /* This function generates the array of kVals to work on and acts as a wrapper for 
     comparison.
  */
  template<typename T>
  void runTests (uint generateType, char* fileName, uint startPower, uint stopPower
                 , uint timesToTestEachK, uint startK, uint stopK, uint kJump) {
    uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 1};
    uint size, i;

    // double the array size to the next powers of 2
    for(size = (1 << startPower); size <= (1 << stopPower); size <<= 1) {

      for(i = startK; i <= stopK; i+=kJump) {
        cudaDeviceReset();
        cudaThreadExit();
        printf("NOW ADDING ANOTHER K\n\n");
        compareMultiselectAlgorithms<T>(size, i, timesToTestEachK, 
                                        algorithmsToRun, generateType, fileName);
      }
    }
  }
}

#endif
