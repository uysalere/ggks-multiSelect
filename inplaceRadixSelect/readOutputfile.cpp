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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <utility>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <dirent.h>
#include <string>

#include <iostream>
#include <iomanip>
#include <limits>
#include <math.h>

struct lineStruct{
  uint numAlgorithms;
  uint size;
  uint k;
  char distribution[100];
  char* names[10];
  uint identifier[10];
  double ratios[10];
  unsigned long long seed;
  uint errorFlag;
  float times[20];
};

void freeLineStruct(lineStruct *line){
  uint i;
  for(i =0; i < line->numAlgorithms; i++){
    free(line->names[i]);
  }
  free(line);
}

void freeData(lineStruct **tests, uint numTests){
  uint i;
  for(i = 0;i < numTests;i++){
    freeLineStruct(tests[i]);
  }
}

void printLineStruct(lineStruct *line){
  printf("Number of algorithms: %u\n", line->numAlgorithms);
  printf("Size: %u\n", line->size);
  printf("Distribution: %s\n", line->distribution);
  printf("Error flag: %u\n", line->errorFlag);
  uint i;
  for(i = 0; i < line->numAlgorithms; i++){
     printf("Name: %s\n", line->names[i]);
     printf("Time : %f\n", line->times[i]);
  }
}
void getFileInfo(char * fileName, uint &numFields, uint &numLines){
  uint fields = 0;
  uint lines = 1;
  char firstLine[2000];
  char *ptr;
  FILE *fp;
  fp = fopen(fileName,"r");
  fgets(firstLine, 2000, fp);
  ptr = firstLine;
  ptr = strtok(firstLine,",");
  while(ptr != NULL){
    ptr = strtok(NULL,",");
    fields++;
  }
  while(fgets(firstLine,2000,fp) != NULL){
    lines++;
  }
  numFields = fields;
  numLines = lines;
  fclose(fp);
}

uint getIdentifier(char* name){
  if(!(strcmp("Sort and choose", name))){
    return 0;
  }
  else if(!(strcmp("Radix select", name))){
    return 1;
  }
  else if(!(strcmp("Bucket Select", name))){
    return 2;
  }
  else if(! (strcmp("Quick Select", name))){
    return 3;
  }
  else if(! (strcmp("Plane Cutting", name))){
    return 4;
  }
  else if(! (strcmp("LANL Select", name))){
    return 5;
  }
  return 50;
}
lineStruct* readLine(FILE *fp, uint numberOfAlgorithms){
  lineStruct *line;
  line = (lineStruct*) malloc(sizeof(lineStruct));
  char *distribution;
  distribution = (char *) malloc(100 * sizeof(char));
  char* lineText;
 
  char *ptr;
  uint i;
  lineText = (char*) malloc(5000 * sizeof(char));
  char *tempName;
  //set the number of algorithms
  line->numAlgorithms = numberOfAlgorithms;
  //read in the line
  fgets(lineText, 5000, fp);
  //get the size
  ptr =strtok(lineText, ",");
  sscanf(ptr,"%u", &line->size);
  //get the K value
  ptr =strtok(NULL, ",");
  sscanf(ptr,"%u", &line->k);
  ptr =strtok(NULL, ",");
  //get the distribution
  sscanf(ptr,"%[^,]s", &line->distribution);
  //do a dummy strtok to skip the seed
  ptr =strtok(NULL, ",");
  for(i =0; i < numberOfAlgorithms; i++){
    ptr =strtok(NULL, ",");
    tempName = (char*) malloc(100 * sizeof(char));
    strcpy(tempName, ptr);
    line->names[i] = tempName;
    line->identifier[i] = getIdentifier(tempName);
    //do an extra strtok to get past the result as we do not care about this
    ptr =strtok(NULL, ",");
    ptr =strtok(NULL, ",");
    sscanf(ptr,"%f", &line->times[i]);
  }
  for(i = 0; i < numberOfAlgorithms; i++){
	  line->ratios[i] = line->times[i]/ line->times[2];
  }
  ptr =strtok(NULL, ",");
  sscanf(ptr, "%u", &line->errorFlag);

  free(distribution);
  free(lineText);
  return line;
}



lineStruct** gatherTests(char* baseFileName,uint &numTests){
  //the folowign code just creates a set of all the different files that begin
  //with the base name
  DIR *dpdf;
  char* tempFn;
  struct dirent *epdf;
  std::set<char*> filesToLookAt;
  std::set<char*>::iterator it;
  dpdf = opendir("./");
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
      if(! (strncmp(epdf->d_name, baseFileName, strlen(baseFileName)))){
        tempFn = (char*) malloc(100 * sizeof(char));
        strcpy(tempFn, epdf->d_name);
        filesToLookAt.insert(tempFn);
      }
    }
  }

  lineStruct **lines= NULL;
  uint totalLines = 0;
  uint linesRead = 0;
  uint i,j;
  uint numberOfAlgorithms;
  uint fieldCount = 0;
  uint numLines = 1;
  lineStruct *temp;

  FILE *fp;

  //for each of the files open it, alloacte more memory, and read it in
  for(i=0,it = filesToLookAt.begin(); it != filesToLookAt.end();it++,i++){
    printf("%s\n", (*it));
    //sprintf(fileName, "%s%u.csv", baseFileName, (i+1) * 100);
    getFileInfo((*it), fieldCount,numLines);
    numberOfAlgorithms = (fieldCount - 5) / 3;
    totalLines += numLines;
    fp = fopen((*it), "r");
    numberOfAlgorithms = (fieldCount - 5) / 3;
    //allocate additional memory
    if(it == filesToLookAt.begin()){
      lines = (lineStruct**) malloc(totalLines * sizeof(lineStruct*));
    }
    else{
      lines = (lineStruct**) realloc(lines, totalLines * sizeof(lineStruct*));
    }
    //read each of the line in and store it in the array
    for(j =0; j < numLines; j++){
      temp = readLine(fp, numberOfAlgorithms);
      lines[linesRead] = temp;
      linesRead++;
    }
  fclose(fp);

  }

  numTests = totalLines;
  return lines;
}

void printRatios(uint size,uint k, lineStruct **tests, uint numTests, uint numberOfAlgorithms, char *outputFile){
	double maxRatios[numberOfAlgorithms];
	double minRatios[numberOfAlgorithms];
	double totalRatios[numberOfAlgorithms];
	bzero(totalRatios, numberOfAlgorithms * sizeof(double));
	uint i,j,testsOfThisType = 0;
	double tempRatio;
	for(i =0; i < numberOfAlgorithms;i++){
      minRatios[i] = 100000;
    }
    for(i =0; i < numberOfAlgorithms;i++){
      maxRatios[i] = 0;
    }
	for(i = 0; i < numTests; i++){
    if((tests[i]->size == size) && (tests[i]->k ==k)){
      for(j =0; j < numberOfAlgorithms;j++){
        tempRatio = tests[i]->ratios[j];
        totalRatios[j] += tempRatio;
        if(tempRatio < minRatios[j]){
          minRatios[j] = tempRatio;
        }
        if(tempRatio > maxRatios[j]){
          maxRatios[j] = tempRatio;
        }
      }
      testsOfThisType++;
    }
  }

  std::ofstream outFileStream;
  outFileStream.open(outputFile, std::ios_base::app);
  outFileStream << size << "," << k << ",";
  for(i = 0; i < numberOfAlgorithms; i++){
    outFileStream <<tests[0]->names[i] << "," << totalRatios[i] / testsOfThisType << "," << maxRatios[i] << "," << minRatios[i] << ",";
  }
  outFileStream << "\n";
  outFileStream.close();
}
//this function will print out the summary data for a combination of n and k 
void printSizeKCombination(uint size, uint k, lineStruct **tests, uint numTests, uint numberOfAlgorithms,char *outputFile){
 
  float totalTimes[numberOfAlgorithms];
  float minTimes[numberOfAlgorithms];
  float maxTimes[numberOfAlgorithms];
  uint timesWon[numberOfAlgorithms];
  bzero(totalTimes, numberOfAlgorithms * sizeof(float));
  bzero(maxTimes, numberOfAlgorithms * sizeof(float));
  bzero(timesWon, numberOfAlgorithms * sizeof(uint));
  uint i,j;

  for(i =0; i < numberOfAlgorithms;i++){
      minTimes[i] = 100000;
    }
  
  //calculate the total times each one took
  float tempTime;
  float lowestSoFar;
  uint winnerSoFar =0;
  uint testsOfThisType = 0;
  uint errorsForThisType = 0;
  for(i = 0; i < numTests; i++){
    lowestSoFar = 1000000;
    if((tests[i]->size == size) && (tests[i]->k ==k)){
      errorsForThisType += tests[i]->errorFlag;
      for(j =0; j < numberOfAlgorithms;j++){
        tempTime = tests[i]->times[j];
        totalTimes[j] += tempTime;
        if(tempTime < minTimes[j]){
          minTimes[j] = tempTime;
        }
        if(tempTime > maxTimes[j]){
          maxTimes[j] = tempTime;
        }
        if(tempTime < lowestSoFar){
          lowestSoFar = tempTime;
          winnerSoFar = j;
        }
      }
      timesWon[winnerSoFar]++;
      testsOfThisType++;
    }
  }

  std::ofstream outFileStream;
  outFileStream.open(outputFile, std::ios_base::app);
  outFileStream << size << "," << k << ",";
  for(i = 0; i < numberOfAlgorithms; i++){

    outFileStream <<tests[0]->identifier[i] << "," << totalTimes[i] / testsOfThisType << "," << maxTimes[i] << "," << minTimes[i] << "," << timesWon[i] <<",";
  }
  outFileStream << testsOfThisType << "," << errorsForThisType;
  outFileStream << "\n";
  outFileStream.close();

}

void printRatioOverview(uint size, lineStruct **tests, uint numTests, uint numberOfAlgorithms,char *outputFile){
  //decare variabes and allocate memory, also set initial vales
  double totalRatios[numberOfAlgorithms];
  double minRatios[numberOfAlgorithms];
  double maxRatios[numberOfAlgorithms];
  uint timesWon[numberOfAlgorithms];
  bzero(totalRatios, numberOfAlgorithms * sizeof(double));
  bzero(maxRatios, numberOfAlgorithms * sizeof(double));
  bzero(timesWon, numberOfAlgorithms * sizeof(uint));
  uint i,j;
	for(i =0; i < numberOfAlgorithms;i++){
      minRatios[i] = 100000;
    }
    for(i =0; i < numberOfAlgorithms;i++){
      maxRatios[i] = 0;
    }
  
  //declare variables
  double tempRatio;
  uint testsOfThisType = 0;
 
  
  //fore ach of the tests run
  for(i = 0; i < numTests; i++){
    //if the size of the test matches the size we are looking for include its 
    //values in the statistics reported. 
    if(tests[i]->size == size){
      for(j =0; j < numberOfAlgorithms;j++){
        tempRatio = tests[i]->ratios[j];
        totalRatios[j] += tempRatio;
        if(tempRatio < minRatios[j]){
          minRatios[j] = tempRatio;
        }
        if(tempRatio > maxRatios[j]){
          maxRatios[j] = tempRatio;
        }
      }
	testsOfThisType++;
    }
  }
  //open up the output file and write the relevant information
  std::ofstream outFileStream;
  outFileStream.open(outputFile, std::ios_base::app);
  outFileStream << size << ",";
  for(i = 0; i < numberOfAlgorithms; i++){
    outFileStream <<tests[0]->names[i] << "," << totalRatios[i] / testsOfThisType << "," << maxRatios[i] << "," << minRatios[i]<< ",";
  }
  outFileStream << "\n";
  outFileStream.close();
}

void printOverview(uint size, lineStruct **tests, uint numTests, uint numberOfAlgorithms,char *outputFile){
  //decare variabes and allocate memory, also set initial vales
  float totalTimes[numberOfAlgorithms];
  float minTimes[numberOfAlgorithms];
  float maxTimes[numberOfAlgorithms];
  uint timesWon[numberOfAlgorithms];
  bzero(totalTimes, numberOfAlgorithms * sizeof(float));
  bzero(maxTimes, numberOfAlgorithms * sizeof(float));
  bzero(timesWon, numberOfAlgorithms * sizeof(uint));
  uint i,j;

  //yeah this could be done better, but this works
  for(i =0; i < numberOfAlgorithms;i++){
    minTimes[i] = 1000000;
  }

  //declare variables
  float tempTime;
  float lowestSoFar;
  uint winnerSoFar =0;
  uint testsOfThisType = 0;
  uint errorsForThisType = 0;
  
  //fore ach of the tests run
  for(i = 0; i < numTests; i++){
    lowestSoFar = INFINITY;
    //if the size of the test matches the size we are looking for include its 
    //values in the statistics reported. 
    if(tests[i]->size == size){
      errorsForThisType += tests[i]->errorFlag;
      for(j =0; j < numberOfAlgorithms;j++){
        tempTime = tests[i]->times[j];
        totalTimes[j] += tempTime;
        if(tempTime < minTimes[j]){
          minTimes[j] = tempTime;
        }
        if(tempTime > maxTimes[j]){
          maxTimes[j] = tempTime;
        }
        if(tempTime < lowestSoFar){
          lowestSoFar = tempTime;
          winnerSoFar = j;
        }
      }
      timesWon[winnerSoFar]++;
      testsOfThisType++;
    }
  }
  //open up the output file and write the relevant information
  std::ofstream outFileStream;
  outFileStream.open(outputFile, std::ios_base::app);
  outFileStream << size << ",";
  for(i = 0; i < numberOfAlgorithms; i++){
    outFileStream <<tests[0]->identifier[i] << "," << totalTimes[i] / testsOfThisType << "," << maxTimes[i] << "," << minTimes[i] << "," << timesWon[i] <<",";
  }
  outFileStream << testsOfThisType << "," << errorsForThisType;
  outFileStream << "\n";
  outFileStream.close();
}


void printTestsSummary(uint numTests, lineStruct **tests, char *baseFileName){
  char *summaryFileName, *overviewFileName, *ratioOverviewFileName,*ratioSummaryFileName;
  summaryFileName = (char*) malloc(100 * sizeof(char));
  ratioSummaryFileName = (char*) malloc(100 * sizeof(char));
  ratioOverviewFileName = (char*) malloc(100 * sizeof(char));

  overviewFileName = (char*) malloc(100 * sizeof(char));
  //create the names for the output files
  sprintf(summaryFileName, "Summary%s.csv", baseFileName);
  sprintf(overviewFileName, "Overview%s.csv", baseFileName);
  sprintf(ratioSummaryFileName, "RatioSummary%s.csv", baseFileName);
  sprintf(ratioOverviewFileName, "RatioOverview%s.csv", baseFileName);
   
  //open and close the output files, this is a simple way to erase their contents
  std::ofstream outFileStream;
  outFileStream.open(summaryFileName);
  outFileStream.close();
  outFileStream.open(overviewFileName);
  outFileStream.close();
  //create sets one containing all of the  (n,k) pairs
  //another containing just the various valeus of n
  std::set<std::pair<uint,uint> > problemSizeKSet;
  std::set<uint> problemSizeSet;
  std::set<std::pair<uint,uint> >::iterator it;
  std::set<uint>::iterator it2;
  uint i;
  //populate the sets
  for(i =0; i < numTests; i++){
    problemSizeKSet.insert(std::make_pair((tests[i])->size, (tests[i])->k));
    problemSizeSet.insert(tests[i]->size);
  }
  //for each element in the set of (n,k) pairs print out the information for that combination
  for(it = problemSizeKSet.begin(); it != problemSizeKSet.end();it++){
    printSizeKCombination((*it).first, (*it).second, tests,numTests, tests[0]->numAlgorithms,summaryFileName);
    printRatios((*it).first, (*it).second, tests,numTests, tests[0]->numAlgorithms,ratioSummaryFileName);
  }
  //for each element in the set of sizes print out the information for that combination
  for(it2 = problemSizeSet.begin(); it2 != problemSizeSet.end(); it2++){
    printOverview((*it2), tests,numTests, tests[0]->numAlgorithms,overviewFileName);
    printRatioOverview((*it2), tests,numTests, tests[0]->numAlgorithms,ratioOverviewFileName);

  }
}
 


void createTestSummary(char *baseFileName){
  lineStruct **tests;
  uint numTests;
  tests = gatherTests(baseFileName, numTests);
  printTestsSummary(numTests, tests, baseFileName);
  freeData(tests, numTests);
}


int main(int argc, char** argv){
  char *fileName;
  fileName = (char*) malloc(60 * sizeof(char));
  if(argc != 2){
    printf("Please enter base file name: ");
    scanf("%s", fileName);
  }
  else{
    fileName = argv[1];
  }
  createTestSummary(fileName);

  return 0;
}
