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

Authors: Russel Steinbach, Bradley Gordon, Jeffrey Blanchard, and
Toluwaloju Alabi.

GGKS project
All original work done for this project is licensed under the Apache 2.0
license. 

Contents:
  Libraries: a listing of libraries relied upon in the GGKS package
  GGKS Algorithms: a brief description of the algorithms created by GGKS
  Non GGKS algorithms: a brief description of algorithms included in GGKS
                    that come partially or completely from another source.
  Running Tests: provides a guide to compilation and usage of the included
 		 testing environment. 
  Data Summaries: provides a guide on how to create files that summarize the
  data that has been produced from running tests. 
  Users Guide: a description of the users guide that is included

 ************************************************************************
 ************************************************************************
 ********************** LIBRARIES ***************************************
 ************************************************************************
 ************************************************************************

THRUST: a variety of thrust primitives are used throughout GGKS, so it is
necessary to have a copy downloaded. For convenience a copy has been
included. 

GSL: the GNU scientific library is used in the randomized selection
algorithm and thus must be present to use that algorithm.

 ************************************************************************
 ************************************************************************
 **********************  GGKS ALGORITHMS ********************************
 ************************************************************************
 ************************************************************************
 
 RADIX SELECT: The interface for radix select is provided in
 merrilSelect.cu. This is named as such in reference to the fact that it
 relies heavily upon code form Merrill's radix sort. The algorithm is best
 described as a modified most significant digit radix sort. For more
 details see merrilSelectGuide.txt in the users guide folder. 

 INPLACE RADIX SELECT: The interface for inplace radix select is provided
 in inplaceRadidxSelect.cu. It is named as the algorithm is inplace. That
 is, it requires only the amount of space to store the problem vector
 once. Unlike normal radix select, which requires two times the size of the
 problem vector. This means that it can solve problems approximately twice
 the size of those done by radix select, or the sort and choose method. For
 more details see inplaceRadixSelectGuide.txt

 BUCKET SELECT: The interface, and complete code, for bucket select is
 provided in bucketSelect.cu. It is named as we conceptually place elements
 into buckets, determine which bucket contains the kth largest element and
 then recurse until we have identified the kth largest element. It is
 actually divided into two phases, for more details see
 bucketSelectGuide.txt 

 ************************************************************************
 ************************************************************************
 ********************** NON GGKS ALGORITHMS *****************************
 ************************************************************************
 ************************************************************************

OTHER WORK: we also include two algorithms not developed by GGKS,

Cutting Plane: For the sake of comparison we have included Beliakov's
cutting plane algorithm. The implementation was not done by GGKS

For details see the paper:
Beliakov, G. 2011. Parallel calculation of the median and order statistics
on  gpus with application to robust regression. Computing Research
Repository abs/1104.2. 


Randomized Selection: We have also included our implementation of
randomized selection. This is based on the
description of an algorithm given by Monroe, Wendelberger, and
Michalak. For details of see the paper: 

Laura Monroe, Joanne Wendelberger, and Sarah Michalak. 2011. Randomized
selection on the GPU. In Proceedings of the ACM SIGGRAPH Symposium on High
Performance Graphics (HPG '11), Stephen N. Spencer (Ed.). ACM, New York,
NY, USA, 89-98. 

One should note that the implementation is completely the work of GGKS and
instead of selecting the top K elements as the algorithm described by
Monroe et al, our implementation returns only the Kth element. For more
details on our implementation and how it differs from the algorithm
described by Monroe et al. See randomizdedSelectGuide.txt




 ************************************************************************
 ************************************************************************
 ********************* RUNNING TESTS ************************************
 ************************************************************************
 ************************************************************************

Running your own tests:
To run your own test use the file compareAlgorithms.cu. It can be compiled
using:

nvcc -o compareAlgorithms compareAlgorithms.cu  -I./lib/ -I. -arch=sm_20 -lcurand -lm -lgsl -lgslcblas

NOTES

1. It is important to have -arch=sm_20, if your machine is capable. This is
because several algorithms perform faster than if sm_13 is used. All
timings done as part of GGKS were done on GPUs of compute capability > 2.0

2. this compile command may need to be modified to accurately reflect
where libraries are on your system. 

3.: It takes a LONG time for compareAlgorithms to compile. This is due to
a variety of factors. Primarily the dependence on thrust sort. Thrust
sort by it self takes a while to compile. Since we also include
radixSelect and inplaceRadixSelect, each of which relies upon parts of
thrust sort this time is increased. 

When run, compareAlgorithms will prompt the user for information about the
tests. It will first prompt for what data type is to be tested. Note that
in current form uint does not work. This is due to the fact that Cutting
plane algorithm does not work on unsigned ints. If you wish to test
unsigned integers remove the cutting plane algorithm from the algorithms to
be tested. Detailed instructions are in compareAlgorithmsGuide.txt in the
UsersGuide folder.


 ************************************************************************
 ************************************************************************
 ********************** DATA SUMMARIES **********************************
 ************************************************************************
 ************************************************************************
 
 Compare algorithms creates a large amount of data, so it is useful to be
 able to process this data. To this end we provide readOutputfile.cpp. It
 can be compiled with:
     
     gcc -o readOutputfile readOutputfile.cpp -lstdc++ -lm

 When run it will prompt for a "base file name". Base file name refers to a
 feature of readOutputfile. It will search the current directory for any
 file that has the base file name at the beginning. For example if you had
 files "file1.csv, file2.csv, 1file.csv", and you provided the base file
 name of "file", it would creates summaries based on tests in "file1.csv"
 and "file2.csv". This feature is useful if you want incorporate multiple
 sets of tests. It is however required that each file tests the same
 algorithms. 

 Once provided with the base file name, readOutputFile will gather all of
 the tests located in those files. It will then create four files.

 Overview[basefilename].csv
        This contains the most general data, based only on the size of the
        vector. The data is in the following format:

size of problem, algorithm identifier1, average time, maximum time, minimum
time, number of times the algorithm was the fastest, algorithm identifier
2,... total number of tests run of this size, total number of times
algorithms provided different answers(ie one was wrong). 

  
  Summary[basefilename].csv
        This contains some what more detailed information. It breaks down
        the data based both on the size of the problem and on which value
        of k was used. It provides information in the same format as
        Overview does except that there is an additional field indicating
        which value of K the row is summarizing. 

  
  RatioOverview[basefilename].csv:
        this provides a ratio of times between the algorithms and bucket
        select, with tests grouped by size.

  RatioSummary[basefilename].csv:
        The same information as RatioOverview except the tests are grouped
        by both size and k. 

 ************************************************************************
 ************************************************************************
 **********************  USERS GUIDE ************************************
 ************************************************************************
 ************************************************************************

 The users guide files, contained in the folder UsersGuide, provide a fair
 amount of commentary on the GGKS package. In general they describe the
 contents of the important files and if applicable how to use them and in
 some cases how to modify them in useful ways. 

 An important part of the users guide is the "Code Overview" section. The
 goal of these are to provide a thorough description of the code in the
 corresponding file. It not only describes what the code accomplishes but
 also some of the rational behind why it was done that way. I have tried to
 be as thorough as possible, and hopefully it will answer any questions you
 may have. Not all of files have a users guide, this is because some of the
 files are either self explanatory of simply not important enough.  

R.S.
