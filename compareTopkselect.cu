/* Copyright 2012 Jeffrey Blanchard, Erik Opavsky, and Emircan Uysaler
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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

// include compareMultiselect library
#include "compareTopkselect.hpp"

/* main fucntion that takes user input to run compare Multiselect on
 */

int main (int argc, char *argv[]) {

  char *fileName, *hostName, *typeString;

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

  uint testCount,type,distributionType,startPower,stopPower,startK,stopK,jumpK;
  
  printf("Please enter the type of value you want to test:\n0-float\n1-double\n2-uint\n");
  scanf("%u", &type);
  printf("Please enter distribution type: ");
  printDistributionOptions(type);
  scanf("%u", &distributionType);
  printf("Please enter Start power: ");
  scanf("%u", &startPower);
  printf("Please enter Stop power: ");
  scanf("%u", &stopPower); 
  printf("Please enter Start number of top-K values: ");
  scanf("%u", &startK);
  printf("Please enter number of top-K values to jump by: ");
  scanf("%u", &jumpK);
  printf("Please enter Stop number of top-K values: ");
  scanf("%u", &stopK);
  printf("Please enter number of tests to run per top-K: ");
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

  snprintf(fileName, 128, 
           "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
           typeString, getDistributionOptions(type, distributionType), 
           " top-K ", startPower, stopPower, 
           startK, jumpK, stopK, testCount, hostName, humanTime);
  printf("File Name: %s \n", fileName);

  using namespace CompareTopkselect;

  switch(type){
  case 0:
    runTests<float>(distributionType,fileName,startPower,stopPower,testCount,
                    startK,stopK,jumpK);
    break;
  case 1:
    runTests<double>(distributionType,fileName,startPower,stopPower,testCount,
                     startK,stopK,jumpK);
    break;
  case 2:
    runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,
                   startK,stopK,jumpK);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }

  free (fileName);
  return 0;
}

