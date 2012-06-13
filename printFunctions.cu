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

/*this file is not used very much in the rest of the program. Infact 
 * the only time they are called is to print out the error result in 
 * binary. It is included mostly as a conveniance. They can be very 
 * helpful in debugging/ watching the steps of an alogrithm.*/
#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>
#include <iostream>
#include <typeinfo>




namespace PrintFunctions{
 
  union udub{
    double d;
    unsigned long long ull;
  };
  union uf{
    float f;
    unsigned int u;
  };
 
  template<typename T>
  void printArray(T *h_vec,uint size){
    printf("\n");
    for(int i = 0; i < size; i++){
      std::cout <<h_vec[i] << "\n";
    }
    printf("\n");
  }
  
  void printArray(char **h_vec,uint size){
    printf("\n");
    for(int i = 0; i < size; i++){
      printf("%d-%s\n",i, h_vec[i]);
    }
    printf("\n");
  }

  template<typename T>
  void printCudaArray(T *d_vec,uint size){
    T* h_vec = (T *) std::malloc(size * sizeof(T));
    cudaMemcpy(h_vec, d_vec, size*sizeof(T), cudaMemcpyDeviceToHost);
    printArray(h_vec, size);
    free(h_vec);
  }

  void printBinary(uint input){
    for(int i = 0; i < 32; i++){
      if(! (i % 4)){
        printf("|");
      }
      printf("%u",(input >> (32 - 1 - i)) & 0x1);
    }
    printf("\n");
  }


  void printBinary(float input){
    uf bits;
    bits.f= input;
    for(int i = 0; i < 32; i++){
      if(! (i % 4)){
        printf("|");
      }
      printf("%u",(bits.u >> (32 - 1 - i)) & 0x1);
    }
    printf("\n");
  }
  
  void printBinary(double input){
    udub bits;
    bits.d = input;
    for(int i = 0; i < 64; i++){
      if(! (i % 4)){
        printf("|");
      }
      printf("%u",(bits.ull >> (64 - 1 - i)) & 0x1);
    }
    printf("\n");
  }

  void printBinary(unsigned long long input){
    for(int i = 0; i < 64; i++){
      if(! (i % 4)){
        printf("|");
      }
      printf("%u",(input >> (64 - 1 - i)) & 0x1);
    }
    printf("\n");
  }


  template<typename T>
  void printArrayBinary(T *h_vec,uint size){  
    printf("\n");
    for(int i = 0; i < size; i++){
      printf("%u: ", i);
      printBinary(h_vec[i]);
    }
    printf("\n");
  }

 
  template<typename T>
  void printCudaArrayBinary(T *d_vec,uint size){
    T* h_vec = (T *) std::malloc(size * sizeof(T));
    cudaMemcpy(h_vec, d_vec, size*sizeof(T), cudaMemcpyDeviceToHost);
    printArrayBinary(h_vec, size);
    free(h_vec);
  }

}//end namespace 
