/* Based on compareAlgorithms.cu */

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
#include "naiveBucketMultiselect.cu"

#include "generateProblems.cu"
#include "multiselectTimingFunctions.cu"

#define NUMBEROFALGORITHMS 3
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] = {"Sort and Choose Multiselect", "Bucket Multiselect", "Naive Bucket Multiselect"};


using namespace std;
template<typename T>
void compareMultiselectAlgorithms(uint size, uint * kVals, uint kListCount, uint numTests, uint *algorithmsToTest, uint generateType, uint kGenerateType, char* fileNamecsv) {
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
 
  typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint *, uint);
  typedef void (*ptrToGeneratingFunction)(T*, uint, curandGenerator_t);

  //these are the functions that can be called
  ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {&timeSortAndChooseMultiselect<T>,
                                                                    &timeBucketMultiselect<T>, 
                                                                    &timeNaiveBucketMultiselect<T>};
  
  ptrToGeneratingFunction *arrayOfGenerators;
  char** namesOfGeneratingFunctions;
  //this is the array of names of functions that generate problems of this type, ie float, double, or uint
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
  printf("The k distribution is: %s\n", namesOfKGenerators[kGenerateType]);

  //*/******************* START RUNNING TESTS *************
  /***********************************************/

  for(i = 0; i < numTests; i++) {
    // cudaDeviceReset();
    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
    
    for(m = 0; m < NUMBEROFALGORITHMS;m++)
      runOrder[m] = m;
    
    std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
    //fileCsv << size << "," << kVals[0] << "," << kVals[kListCount - 1] << "," << kListCount << "," << (100*((float)kListCount/size)) << "," << namesOfGeneratingFunctions[generateType] << "," << namesOfKGenerators[kGenerateType] << "," << seed << ",";
    fileCsv << size << "," << kListCount << "," << namesOfGeneratingFunctions[generateType] << "," << namesOfKGenerators[kGenerateType] << ",";
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    printf("Running test %u of %u for size: %u and numK: %u\n", i + 1, numTests, size, kListCount);
    //generate the random vector using the specified distribution
    arrayOfGenerators[generateType](h_vec, size, generator);

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
        temp = arrayOfTimingFunctions[j](h_vec_copy, size, kVals, kListCount);

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

    uint flag = 0;
    for(m = 1; m < NUMBEROFALGORITHMS;m++)
      if(algorithmsToTest[m])
        for (j = 0; j < kListCount; j++) {
          if(resultsArray[m][i][j] != resultsArray[0][i][j]) {
            flag++;
            fileCsv << "\nERROR ON TEST " << i << " of " << numTests << " tests!!!!!\n";
            fileCsv << "vector size = " << size << "\nvector seed = " << seed << "\n";
            fileCsv << "kListCount = " << kListCount << "\n";
            fileCsv << "wrong k = " << kVals[j] << " kIndex = " << j << " wrong result = " << resultsArray[m][i][j] << " correct result = " <<  resultsArray[0][i][j] << "\n";
            std::cout <<namesOfMultiselectTimingFunctions[m] <<" did not return the correct answer on test " << i + 1 << " at k[" << j << "].  It got "<< resultsArray[m][i][j];
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
  free(h_vec);
  free(h_vec_copy);
  //close the file
  fileCsv.close();
}


template<typename T>
void runTests (uint generateType, char* fileName, uint startPower, uint stopPower, uint timesToTestEachK, 
               uint kDistribution, uint startK, uint stopK, uint kJump) {
  uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 0};
  uint size;
  uint i;
  uint arrayOfKs[stopK+1];
  
  
  for(size = (1 << startPower); size <= (1 << stopPower); size *= 2) {
    /*
    //calculate k values
    arrayOfKs[0] = 2;
    //  arrayOfKs[1] = (uint) (.01 * (float) size);
    //  arrayOfKs[2] = (uint) (.025 * (float) size);
    for(i = 1; i <= num - 2; i++) 
    arrayOfKs[i] = (uint) (( i / (float) num ) * size);
    
    //  arrayOfKs[num-3] = (uint) (.9975 * (float) size);
    //  arrayOfKs[num-2] = (uint) (.999 * (float) size);
    arrayOfKs[num-1] = (uint) (size - 2); 
    */
    unsigned long long seed;
    timeval t1;
    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
    curandGenerator_t generator;
    srand(unsigned(time(NULL)));
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);

    arrayOfKDistributionGenerators[kDistribution](arrayOfKs, stopK, size, generator);

    curandDestroyGenerator(generator);

    /*
    printf("arrayOfKs = ");
    for(uint j = 0; j < stopK+1; j++)
      printf("%u; ", arrayOfKs[j]);
    printf("\n\n");
    */

    for(i = startK; i <= stopK; i+=kJump) {
      cudaDeviceReset();
      cudaThreadExit();
      printf("NOW ADDING ANOTHER K\n\n");
      compareMultiselectAlgorithms<T>(size, arrayOfKs, i, timesToTestEachK, algorithmsToRun, generateType, kDistribution, fileName);
    }
  }
}


int main (int argc, char *argv[]) {
  char *fileName, *hostName, *typeString;

  uint testCount;
  fileName = (char*) malloc(128 * sizeof(char));
  typeString = (char*) malloc(10 * sizeof(char));
  hostName = (char*) malloc(20 * sizeof(char));
  gethostname(hostName, 20);

  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  char * humanTime = asctime(timeinfo);
  humanTime[strlen(humanTime)-1] = '\0';

  uint type,distributionType,startPower,stopPower,kDistribution,startK,stopK,jumpK;
  
  printf("Please enter the type of value you want to test:\n0-float\n1-double\n2-uint\n");
  scanf("%u", &type);
  printf("Please enter distribution type: ");
  printDistributionOptions(type);
  scanf("%u", &distributionType);
  printf("Please enter K distribution type: ");
  printKDistributionOptions();
  scanf("%u", &kDistribution);
  printf("Please enter Start power: ");
  scanf("%u", &startPower);
  printf("Please enter Stop power: ");
  scanf("%u", &stopPower); 
  printf("Please enter Start number of K values: ");
  scanf("%u", &startK);
  printf("Please enter number of K values to jump by: ");
  scanf("%u", &jumpK);
  printf("Please enter Stop number of K values: ");
  scanf("%u", &stopK);
  printf("Please enter number of tests to run per K: ");
  scanf("%u", &testCount);

  switch(type){
  case 0:
    typeString = "float";
    break;
  case 1:
    typeString = "double";
    break;
  case 2:
    typeString = "uint";
    break;
  default:
    break;
  }

  snprintf(fileName, 128, "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", typeString, getDistributionOptions(type, distributionType), getKDistributionOptions(kDistribution), startPower, stopPower, startK, jumpK, stopK, testCount, hostName, humanTime);
  printf("File Name: %s \n", fileName);
  //printf("Please enter filename now: ");

  switch(type){
  case 0:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK);
    break;
  case 1:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK);
    break;
  case 2:
    runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }

  free (fileName);
  return 0;
}
