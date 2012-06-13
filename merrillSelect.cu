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

/* NOTE: The following algorithms depend upon a modifed version of Merrill's
 * Radix Sort algorithm. Parts of some functions are identical to that in 
 * in file thrust/detail/device/cuda/detail/stable_radix_sort_merrill.inl
 */

#include <RadixSelect/RadixSelect_api.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
//Include various thrust items that are used
#include <thrust/detail/util/align.h>



#include "inplaceRadixSelect.cu"
#define RADIX_CUTOFF 1 <<21
namespace RadixSelect
{

template<typename T>
void postProcess(uint *result ){}

template<>
void postProcess<float>(uint *result){
  unsigned int mask = (result[0] & 0x80000000) ?  0x80000000 : 0xffffffff ; 
  result[0] ^= mask; 
}


template<typename T>
void postProcess(unsigned long long *result){}

template<>
void postProcess<double>(unsigned long long *result){
  const unsigned long long mask = (result[0] & 0x8000000000000000) ? 0x8000000000000000 : 0xffffffffffffffff; 
  result[0] ^= mask;
}




template<typename RandomAccessIterator, typename T>
void merrillSelect(RandomAccessIterator first,
                   RandomAccessIterator last,
                   uint k,uint pass,T *result, uint needToPreprocess, RandomAccessIterator temp_keys){
  uint num_elements = last - first;
  if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*first), 2*sizeof(T)))
    {
      cudaMemcpy(thrust::raw_pointer_cast(&*temp_keys),thrust::raw_pointer_cast(&*first), num_elements * sizeof(T), cudaMemcpyDeviceToDevice);
      merrillSelect(temp_keys,temp_keys + num_elements,k, pass,result,needToPreprocess,&*first);
      return;
    }
 

  RadixSelect::RadixSortingEnactor<T> sorter(num_elements);
  RadixSelect::RadixSortStorage<T> storage(k,num_elements,needToPreprocess);
  
  // allocate temporary buffers

  thrust::detail::raw_cuda_device_buffer<int>          temp_spine(sorter.SpineElements());
  thrust::detail::raw_cuda_device_buffer<bool>         temp_from_alt(2);

  //copy the pointers to storage
  storage.d_keys             = thrust::raw_pointer_cast(&*first);
  storage.d_alt_keys          = thrust::raw_pointer_cast(&*temp_keys);
  storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
  storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);

  uint retval = sorter.EnactSort(storage,pass);

  //num_elements is now the number of elements in the new list we are interested in
  num_elements = storage.h_useful[4];

  //if there are stil more passes to go, and there is more than one element that could be 
  //the kth largest element call merrilSelect that will look at the next four bits
  if(pass < ((sizeof(T) * 2) -1) && num_elements > 1){
    //if the elements of the list were not redistributed then pass the inputs to this
    //pass to the next pass, except increment pass by one. 
    if(retval){
      merrillSelect(first,last, k, pass + 1,result, needToPreprocess,temp_keys);
    }
    //otherwise the new list we are interested in is in temp_keys after being scattered 
    //we calculate the new start and stop values by adding the new start index h_useful[2] to
    //begining of temp_keys, the new value of k that we are looking for is
    //in h_useful[7], Additionally since one pass has already been run we know the list has been preprocessed so we should not 
    //preprocess again. 
    else{
      merrillSelect(temp_keys + storage.h_useful[2], temp_keys + storage.h_useful[2] + num_elements, storage.h_useful[7], pass + 1,result,0,first);
    }
  }
  //if we do not need to do another pass then we just copy the result back to the cpu, and call the postprocess function
  else{
    //if the size has not changed then we know the results will still be in the input, so grab the value from there
    if(retval){
      cudaMemcpy(result,thrust::raw_pointer_cast(&*first), 1 * sizeof(T), cudaMemcpyDeviceToHost);
      if(! needToPreprocess){
        postProcess<T>((uint*)result);
      }
      return;
    }
    //otherwise we grab the value from temp_keys since they have been scattered there. 
    else{
      cudaMemcpy(result,thrust::raw_pointer_cast(&*temp_keys)+storage.h_useful[2], 1 * sizeof(T), cudaMemcpyDeviceToHost);    
      postProcess<T>((uint *)result);
      return;
    }
  }

}


 


template<typename RandomAccessIterator>
void merrillSelect(RandomAccessIterator first,
                   RandomAccessIterator last,
                   uint k,uint pass,double *result, uint needToPreprocess,RandomAccessIterator temp_keys){

  uint num_elements = last - first;
  typedef typename thrust::iterator_value<RandomAccessIterator>::type K;

  if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*first), 2*sizeof(K)))
    {
      cudaMemcpy(thrust::raw_pointer_cast(&*temp_keys),thrust::raw_pointer_cast(&*first), num_elements * sizeof(double), cudaMemcpyDeviceToDevice);
      merrillSelect(temp_keys,temp_keys + num_elements,k, pass,result,needToPreprocess,&*first);
      return;
    }
  RadixSelect::RadixSortingEnactor<K> sorter(num_elements);
  RadixSelect::RadixSortStorage<K> storage(k,num_elements,needToPreprocess);
  
  
  // allocate temporary buffers
  thrust::detail::raw_cuda_device_buffer<int>          temp_spine(sorter.SpineElements());
  thrust::detail::raw_cuda_device_buffer<bool>         temp_from_alt(2);

  //copy the pointers to storage
  storage.d_keys             = thrust::raw_pointer_cast(&*first);
  storage.d_alt_keys          = thrust::raw_pointer_cast(&*temp_keys);

  storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
  storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);


  uint retval = sorter.EnactSort(storage,pass);
 
  //num_elements is now the number of elements in the new list we are interested in
  num_elements = storage.h_useful[4];

  //if there are stil more passes to go, and there is more than one element that could be 
  //the kth largest element call merrilSelect that will look at the next four bits
  if(pass < 15 && num_elements > 1){
    //if the elements of the list were not redistributed then pass the inputs to this
    //pass to the next pass, except increment pass by one. 
    if(retval){
      merrillSelect(first,last, k, pass + 1,result, needToPreprocess, temp_keys);
    }
    //otherwise the new list we are interested in is in temp_keys after being scattered 
    //we calculate the new start and stop values by adding the new start index h_useful[2] to
    //begining of temp_keys, the new value of k that we are looking for is
    //in h_useful[7], Additionally since one pass has already been run we know the list has been preprocessed so we should not 
    //preprocess again. 
    else{
      merrillSelect(temp_keys + storage.h_useful[2], temp_keys + storage.h_useful[2] + num_elements, storage.h_useful[7], pass + 1,result,0,first);
    }
  }
  //if we do not need to do another pass then we just copy the result back to the cpu, and call the postprocess function
  else{
    //if the size has not changed then we know the results will still be in the input, so grab the value from there
    if(retval){
      cudaMemcpy(result,thrust::raw_pointer_cast(&*first), 1 * sizeof(double), cudaMemcpyDeviceToHost);
      if(! needToPreprocess){
        postProcess<double>((unsigned long long*)result);
      }
      return;
    }
    //otherwise we grab the value from temp_keys since they have been scattered there. 
    else{
      cudaMemcpy(result,thrust::raw_pointer_cast(&*temp_keys)+storage.h_useful[2], 1 * sizeof(double), cudaMemcpyDeviceToHost);    
      postProcess<double>((unsigned long long *)result);
      return;
    }
  }
  
}

  


  uint RadixSelectWrapper(uint* d_vec,uint size, uint k){
    uint result; 
    uint *temp_keys;
    thrust::device_ptr<uint> dev_ptr(d_vec);
    if(size < (1 << 21)){
      result = InplaceRadix::inplaceRadixSelectWrapper(d_vec, size,k);
    }
    else{
      cudaMalloc(&temp_keys, size * sizeof(uint));
      thrust::device_ptr<uint> dev_temp_ptr(temp_keys);
      merrillSelect(dev_ptr, dev_ptr + size,k, 0, &result,1,dev_temp_ptr);
      cudaFree(temp_keys);
    }
    return result;
  }  

  float RadixSelectWrapper(float* d_vec,uint size, uint k){
    float result;
    float *temp_keys;
    if(size < (1 << 21)){
      result = InplaceRadix::inplaceRadixSelectWrapper(d_vec, size,k);
    }
    else{
      cudaMalloc(&temp_keys, size * sizeof(float));
      thrust::device_ptr<float> dev_ptr(d_vec); 
      thrust::device_ptr<float> dev_temp_ptr(temp_keys); 
      merrillSelect(dev_ptr, dev_ptr + size,k, 0, &result,1,dev_temp_ptr);
      cudaFree(temp_keys);
    }
    return result;
  }  



  double RadixSelectWrapper(double* d_vec,uint size, uint k){
    double result;
    double *temp_keys;
    if(size < (1 << 21)){
      result = InplaceRadix::inplaceRadixSelectWrapper(d_vec, size,k);
    }
    else{
      cudaMalloc(&temp_keys, size * sizeof(double));
      thrust::device_ptr<double> dev_ptr(d_vec); 
      thrust::device_ptr<double> dev_temp_ptr(temp_keys); 
      merrillSelect(dev_ptr, dev_ptr + size,k, 0, &result,1,dev_temp_ptr);
      cudaFree(temp_keys);
    }
    return result;
     
  }
}



