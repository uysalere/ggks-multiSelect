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

#include <thrust/detail/config.h>
#include <inplaceRadix/inplaceRadixSelect_api.h>
#include "keyConversion.cu"

namespace InplaceRadix{


  //Based on stable_radix_sort, in file thrust/detail/device/cuda/detail/stable_radix_sort_merrill.inl
 template<typename RandomAccessIterator, typename T>
 void inplaceRadixSelect(RandomAccessIterator first, RandomAccessIterator last, uint k, T &result)
{
    uint i ;
    typedef typename thrust::iterator_value<RandomAccessIterator>::type K;
    uint num_elements = last - first;
    uint indexOfK = num_elements - k;
    // ensure data is properly aligned
    if (!thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&*first), 2*sizeof(K)))
    {
        thrust::detail::raw_cuda_device_buffer<K> aligned_keys(first, last);
        inplaceRadixSelect(aligned_keys.begin(), aligned_keys.end(), k,result);
        thrust::copy(aligned_keys.begin(), aligned_keys.end(), first);
        return;
    }
    
    InplaceRadix::RadixSortingEnactor<K> sorter(num_elements);
    InplaceRadix::RadixSortStorage<K>    storage(num_elements, indexOfK);
    
    // allocate temporary buffers
    thrust::detail::raw_cuda_device_buffer<int>          temp_spine(sorter.SpineElements());
    thrust::detail::raw_cuda_device_buffer<bool>         temp_from_alt(2);

    // define storage
    storage.d_keys             = thrust::raw_pointer_cast(&*first);
    storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
    storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);
    cudaMalloc(&(storage.d_radix), sizeof(uint));
    // perform the sort

    sorter.EnactSort(storage);

    uint tmp = 0;
    unsigned long long tmpBig = 0;
    //depending on the size of the items in the list either
    //put it into an uint, or an unsigned long long, this is done because you cannot 
    //do bit operations on floating point numbers
    if(sizeof(T) == sizeof(uint)){
      //this retrieves the radixes from storage
      for(i = 0; i < sizeof(T)*2;i++){
        tmp |= (storage.radixes[i] <<(((sizeof(T)*8) - 4)-( 4 * i)));
      }
      memcpy(&result, &tmp, sizeof(uint));
    }
    else{
      unsigned long long tmp2 =0;
      for(i = 0; i < sizeof(double)*2;i++){
        tmp2 = storage.radixes[i];
        tmpBig |= (tmp2 <<(60-( 4 * i)));
      }
      
      memcpy(&result, &tmpBig, sizeof(double));

    }

    // temporary storage automatically freed
} 

	
uint inplaceRadixSelectWrapper(uint *d_vec, uint size, uint k){
  uint result;
  thrust::device_ptr<uint> dev_ptr(d_vec);
  inplaceRadixSelect(dev_ptr, dev_ptr + size,k,result);
  return result;
}

float inplaceRadixSelectWrapper(float *d_vec, uint size, uint k){
  float result;
  preProcess<float> pre;
  thrust::device_ptr<uint> dev_ptrNew((uint *)d_vec);
  thrust::for_each(dev_ptrNew, dev_ptrNew + size,pre);
  inplaceRadixSelect(dev_ptrNew, dev_ptrNew + size,k,result);
  postProcess<float>((uint *) &result);
  return result;
}

double inplaceRadixSelectWrapper(double *d_vec, uint size, uint k){
  double result;
  preProcess<double> pre;
  thrust::device_ptr<unsigned long long> dev_ptrNew((unsigned long long *)d_vec);
  thrust::for_each(dev_ptrNew, dev_ptrNew + size,pre);
  inplaceRadixSelect(dev_ptrNew, dev_ptrNew + size,k,result);
  postProcess<double>((unsigned long long *) &result);
  return result;
}

}
