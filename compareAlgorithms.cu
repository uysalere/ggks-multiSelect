/* Copyright 2011 Russel Steinbach, Jeffrey Blanchard, Bradley Gordon,
 *   and Toluwaloju Alabi
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

//various functions, include the functions
//that print numbers in binary.
#include "printFunctions.cu"

//the algorithms
#include "cp_select.cu"
#include "merrillSelect.cu"
#include "bucketSelect.cu"
#include "randomizedSelect.cu"

#include "generateProblems.cu"
#include "timingFunctions.cu"

#define NUMBEROFALGORITHMS 5
char* namesOfTimingFunctions[NUMBEROFALGORITHMS] = {"Sort and choose", "Radix select","Bucket Select","Plane Cutting","Randomized Select"}; 
 
using namespace std;
template<typename T>
void compareAlgorithms(uint size, uint k, uint numTests,uint *algorithmsToTest, uint generateType,char* fileNamecsv){
  T *h_vec, *h_vec_copy;
  float timeArray[NUMBEROFALGORITHMS][numTests];
  T resultsArray[NUMBEROFALGORITHMS][numTests];
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
  ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {&timeSortAndChoose<T>,&timeRadixSelect<T>, &timeBucketSelect<T>, &timeCuttingPlane<T>, &timeRandomizedSelect<T>};
  
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
  for(i = 0; i < numTests; i++){
    // cudaDeviceReset();
    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
    
    for(m = 0; m < NUMBEROFALGORITHMS;m++){
      runOrder[m] = m;
    }
    std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
    fileCsv << size <<"," << k << "," << namesOfGeneratingFunctions[generateType] << "," << seed<< ",";
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    printf("Running test %u of %u for size: %u and k: %u\n", i + 1, numTests,size,k);
    //generate the random vector using the specified distribution
    arrayOfGenerators[generateType](h_vec, size,generator);

    //copy the vector to h_vec_copy, which will be used to restore it later
    memcpy(h_vec_copy, h_vec,size * sizeof(T));

    winnerArray[i] = 0;
    float currentWinningTime = INFINITY;
    //run the various timing functions
    for(x = 0; x < NUMBEROFALGORITHMS; x++){
      j = runOrder[x];
      if(algorithmsToTest[j]){

        //run timing function j
        printf("TESTING: %u\n", j);
        temp = arrayOfTimingFunctions[j](h_vec_copy,size,k);

        //record the time result
        timeArray[j][i] = temp->time;
        //record the value returned
        resultsArray[j][i] = temp->val;
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
    for(x = 0; x < NUMBEROFALGORITHMS; x++){
      if(algorithmsToTest[x]){
        fileCsv << namesOfTimingFunctions[x] << ","<< resultsArray[x][i] <<","<< timeArray[x][i] <<",";
      }
    }
    uint flag = 0;
    T tempResult = resultsArray[0][i];
    for(m = 1; m < NUMBEROFALGORITHMS;m++){
      if(algorithmsToTest[m]){
        if(resultsArray[m][i] != tempResult){
          flag++;
        }
      }
    }
    fileCsv << flag << "\n";
  }
  
  //calculate the total time each algorithm took
  for(i = 0; i < numTests; i++){
    for(j = 0; j < NUMBEROFALGORITHMS;j++){
      if(algorithmsToTest[j]){
        totalTimesPerAlgorithm[j] += timeArray[j][i];
      }
    }
  }

  //count the number of times each algorithm won. 
  for(i = 0; i < numTests;i++){
    timesWon[winnerArray[i]]++;
  }

  printf("\n\n");

  //print out the average times
  for(i = 0; i < NUMBEROFALGORITHMS; i++){
    if(algorithmsToTest[i]){
      printf("%-20s averaged: %f ms\n", namesOfTimingFunctions[i], totalTimesPerAlgorithm[i] / numTests);
    }
  }
  for(i = 0; i < NUMBEROFALGORITHMS; i++){
    if(algorithmsToTest[i]){
      printf("%s won %u times\n", namesOfTimingFunctions[i], timesWon[i]);
    }
  }
  for(i = 0; i < numTests; i++){
    for(j =1 ; j< NUMBEROFALGORITHMS; j++){
      if(algorithmsToTest[j]){
        if(resultsArray[j][i] != resultsArray[0][i]){
          std::cout <<namesOfTimingFunctions[j] <<" did not return the correct answer on test" << i +1 << " it got "<< resultsArray[j][i];
          std::cout << "instead of " << resultsArray[0][i] << "\n" ;
          std::cout << "RESULT:\t";
          PrintFunctions::printBinary(resultsArray[j][i]);
          std::cout << "Right:\t";
          PrintFunctions::printBinary(resultsArray[0][i]);
        }
      }
    }
  }
  //free h_vec and h_vec_copy
  free(h_vec);
  free(h_vec_copy);
  //close the file
  fileCsv.close();

}


template<typename T>
void runTests(uint generateType, char* fileName,uint startPower, uint stopPower, uint timesToTestEachK = 100){
  uint algorithmsToRun[NUMBEROFALGORITHMS]= {1,1,1,0,0};
  uint size;
  uint i;
  uint arrayOfKs[25];
  for(size = (1 << startPower); size <= (1 <<stopPower);size *= 2){
    //calculate k values
    arrayOfKs[0]= 2;
    arrayOfKs[1] = .01 * size;
    arrayOfKs[2] = .025 * size;
    for(i = 1; i <= 19; i++){
      arrayOfKs[i +2] = (.05 * i) * size;
    }
    arrayOfKs[22] = .975 * size;
    arrayOfKs[23] = .99 * size;
    arrayOfKs[24] = size-1;

    for(i = 0; i <25; i++){
      //  cudaDeviceReset();
      cudaThreadExit();
      printf("NOW STARTING A NEW K\n\n"); 
      compareAlgorithms<T>(size, arrayOfKs[i], timesToTestEachK,algorithmsToRun,generateType,fileName);
    }
  }
}


int main(int argc, char** argv)
{
  char *fileName;

  uint testCount;
  fileName = (char*) malloc(60 * sizeof(char));
  printf("Please enter filename now: ");
  scanf("%s%",fileName);

  uint type,distributionType,startPower,stopPower;
  
  printf("Please enter the type of value you want to test:\n1-float\n2-double\n3-uint\n");
  scanf("%u", &type);
  printf("Please enter Distribution type: ");
  scanf("%u", &distributionType);
  printf("Please enter  number of tests to run per K: ");
  scanf("%u", &testCount);
  printf("Please enter Start power: ");
  scanf("%u", &startPower);
  printf("Please enter Stop power: ");
  scanf("%u", &stopPower); 

  switch(type){
  case 1:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount);
    break;
  case 2:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount);
    break;
  case 3:
    // runTests<uint>(distributionType,fileName,startPower,stopPower,testCount);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }
  return 0;
}
