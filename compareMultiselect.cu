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
#include <iomanip>
#include <unistd.h>

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
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] = {"Sort and Choose Multiselect", "Bucket Multiselect", "Stable Sort and Choose Multiselect"};

union uf{
  float f;
  unsigned int u;
};

using namespace std;
template<typename T>
void compareMultiselectAlgorithms(uint size, uint * kVals, uint kListCount, uint numTests, uint *algorithmsToTest, uint generateType, uint kGenerateType, char* fileNamecsv, time_t srandSeed, unsigned long long originalSeed, uint mainSeed, time_t generatorSeed, unsigned long long seed) {
  T *h_vec, *h_vec_copy;
  float timeArray[NUMBEROFALGORITHMS][numTests];
  T * resultsArray[NUMBEROFALGORITHMS][numTests];
  float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
  uint winnerArray[numTests];
  uint timesWon[NUMBEROFALGORITHMS];
  uint i,j,m,x;
  int runOrder[NUMBEROFALGORITHMS];

  results_t<T> *temp;
  ofstream fileCsv;
  //timeval t1;
 
  typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint *, uint, uint *);
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
  // time_t generatorSeed = time(NULL);
  srand (unsigned (generatorSeed));

  printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
  printf("The k distribution is: %s\n", namesOfKGenerators[kGenerateType]);
  for(i = 0; i < numTests; i++) {
    // cudaDeviceReset();
    // gettimeofday(&t1, NULL);
    // seed = t1.tv_usec * t1.tv_sec;
    
    for(m = 0; m < NUMBEROFALGORITHMS;m++)
      runOrder[m] = m;
    
    std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
    //  fileCsv << size << "," << kVals[0] << "," << kVals[kListCount - 1] << "," << kListCount << "," << (100*((float)kListCount/size)) << "," << namesOfGeneratingFunctions[generateType] << "," << namesOfKGenerators[kGenerateType] << ", originalSeed " << originalSeed << ", generatorSeed " << generatorSeed << ", seed " << seed << ",";
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    printf("Running test %u of %u for size: %u and numK: %u\n", i + 1, numTests, size, kListCount);
    //generate the random vector using the specified distribution
    arrayOfGenerators[generateType](h_vec, size, generator);

    // print the vector
    /*
    ofstream vectorFileCsv;
    uf bits;
    vectorFileCsv.open("vector3.csv", ios_base::app);
    for (int z = 0; z < size; z++) {
      float f = *(h_vec + z);
      vectorFileCsv << "\'";
      bits.f= f;
      for(int i = 0; i < 32; i++){
      /*
      if(! (i % 4)){
      vectorFileCsv << "|";
      }
    */
    /*
      vectorFileCsv << ((bits.u >> (32 - 1 - i)) & 0x1);
      }
      vectorFileCsv << "\'\n";
      }
      
      vectorFileCsv.close();
    */
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
        //  printf ("kVals[72] = %u\n", kVals[72]);
        //  kVals[71] = 65933979;
        //  kVals[73] = 65933981;
        temp = arrayOfTimingFunctions[j](h_vec_copy, size, kVals, kListCount, &mainSeed);
        printf ("%u, %u\n", kVals[71], kVals[73]);
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

    // check if results are correct
    uint flag = 0;
    for(m = 1; m < NUMBEROFALGORITHMS;m++)
      if(algorithmsToTest[m])
        for (j = 0; j < kListCount; j++) {
          T tempResult = resultsArray[0][i][j];
          if(resultsArray[m][i][j] != tempResult)
            flag++;
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

  for(i = 0; i < numTests; i++)
    for(j = 1; j < NUMBEROFALGORITHMS; j++)
      for (m = 0; m < kListCount; m++)
        if(algorithmsToTest[j])
          if(resultsArray[j][i][m] != resultsArray[0][i][m]) {

            std::cout << "wrong k:" << kVals[354] << ".\n";


            std::cout <<namesOfMultiselectTimingFunctions[j] <<" did not return the correct answer on test " << i + 1 << " at k[" << m << "].  It got "<< resultsArray[j][i][m];
            std::cout << " instead of " << resultsArray[0][i][m] << ".\n" ;
            std::cout << "RESULT:\t";
            PrintFunctions::printBinary(resultsArray[j][i][m]);
            std::cout << "Right:\t";
            PrintFunctions::printBinary(resultsArray[0][i][m]);
            std::cout << size << "," << kVals[0] << "," << kVals[kListCount - 1] << "," << kListCount << "," << (100*((float)kListCount/size)) << "," << namesOfGeneratingFunctions[generateType] << "," << namesOfKGenerators[kGenerateType] << ", originalSeed " << originalSeed << ", generatorSeed " << generatorSeed << ", seed " << seed << ", mainSeed " << mainSeed << "\n";

            // print to file
            fileCsv << "\nERROR!!! on" << namesOfMultiselectTimingFunctions[j] << " did not return the correct answer on test " << i + 1 << " at k[" << m << "]=" << kVals[m] << "\n" ; 
            fileCsv << size << "," << kVals[0] << "," << kVals[kListCount - 1] << ", kListCount = " << kListCount << "," << (100*((float)kListCount/size)) << "," << namesOfGeneratingFunctions[generateType] << "," << namesOfKGenerators[kGenerateType] << ", originalSeed " << originalSeed << ", generatorSeed " << generatorSeed << ", seed " << seed << ", mainSeed " << mainSeed << "\n";
            
          }

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
void runTests (uint generateType, char* fileName, uint startPower, uint stopPower, uint timesToTestEachK, uint kDistribution, uint startK, uint stopK, uint kJump, time_t srandSeed, unsigned long long originalSeed, uint mainSeed, time_t generatorSeed, unsigned long long seed) {
  uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 1};
  uint size;
  uint i;
  uint arrayOfKs[stopK+1];
  
  
  for (size = (1 << startPower); size <= (1 << stopPower); size *= 2) {

    // timeval t1;
    // gettimeofday(&t1, NULL);
    // originalSeed = t1.tv_usec * t1.tv_sec;
    curandGenerator_t generator;
    srand(unsigned(srandSeed));
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,originalSeed);

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
      compareMultiselectAlgorithms<T>(size, arrayOfKs, i, timesToTestEachK, algorithmsToRun, generateType, kDistribution, fileName, srandSeed, originalSeed, mainSeed, generatorSeed, seed );
    }
  }
}


int main (int argc, char *argv[]) {
  //kDistributionSeed 385095184934058, generatorSeed 1351810014, srandSeed 1351808817, arraySeed 621812329289790, randomSampleSeed 2390577621
  // originalSeed = kDistributionSeed
  unsigned long long kDistributionSeed = 87802604328220; // THIS MUST BE FIXED TO GET ERROR
  time_t generatorSeed = time(NULL);
  //time_t generatorSeed = time(NULL); //1351465169; // THIS CAN BE RANDOM
  time_t srandSeed = time (NULL);
  //time_t srandSeed = time (NULL);//1351465069;  // this can also be random
  // seed = arraySeed
  unsigned long long arraySeed = 1313572239387928; // THIS MUST BE FIXED TO GET ERROR
  // randomSampleSeed = mainSeed
  uint randomSampleSeed = 453328713; // THIS MUST BE FIXED TO GET ERROR

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


  snprintf(fileName, 128, "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", typeString, getDistributionOptions(type, distributionType), getKDistributionOptions(kDistribution), startPower, stopPower, startK, jumpK, stopK, testCount, hostName, asctime(timeinfo));
  printf("File Name: %s \n", fileName);
  //printf("Please enter filename now: ");

  //void runTests (uint generateType, char* fileName, uint startPower, uint stopPower, uint timesToTestEachK, uint kDistribution, uint startK, uint stopK, uint kJump, time_t srandSeed, unsigned long long originalSeed, uint mainSeed, time_t generatorSeed, unsigned long long seed) {
  switch(type){
  case 0:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK, srandSeed, kDistributionSeed, randomSampleSeed, generatorSeed, arraySeed);
    break;
  case 1:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK, srandSeed, kDistributionSeed, randomSampleSeed, generatorSeed, arraySeed);
    break;
  case 2:
    runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,kDistribution,startK,stopK,jumpK, srandSeed, kDistributionSeed, randomSampleSeed, generatorSeed, arraySeed);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }

  free (fileName);
  return 0;
}
