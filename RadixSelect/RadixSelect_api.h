/******************************************************************************
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * 
 * 
 * 
 * ORIGINAL AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 *
 *
 * This file has been modified as part of the Grinnell GPU K selection project
 * to perform selection instead of sorting.
 * all modifications are liscensed under the apache 2.0 license
 ******************************************************************************/


#pragma once

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>
#include <iostream>
#include <typeinfo>
#include "thrust/detail/device/cuda/detail/b40c/radixsort_reduction_kernel.h"
#include "thrust/detail/device/cuda/detail/b40c/radixsort_spine_kernel.h"
#include "thrust/detail/device/cuda/detail/b40c/radixsort_scanscatter_kernel.h"

#include <thrust/swap.h>
using namespace thrust::detail::device::cuda::detail::b40c_thrust;
namespace RadixSelect{


/******************************************************************************
 * Debugging options
 ******************************************************************************/

static bool RADIXSORT_DEBUG = false;



/******************************************************************************
 * Structures for mananging device-side sorting state
 ******************************************************************************/

/**
 * Sorting storage-management structure for device vectors
 */
template <typename K, typename V = KeysOnlyType>
struct RadixSortStorage {

  uint *d_useful;
  uint h_useful[8];
  uint *d_storeCounts;
  uint needToPre;
  // Device vector of keys to sort
  K* d_keys;
	
  // Device vector of values to sort
  V* d_values;

  // Ancillary device vector for key storage 
  K* d_alt_keys;

  // Ancillary device vector for value storage
  V* d_alt_values;

  // Temporary device storage needed for radix sorting histograms
  int *d_spine;
	
  // Flip-flopping temporary device storage denoting which digit place 
  // pass should read from which input source (i.e., false if reading from 
  // keys, true if reading from alternate_keys
  bool *d_from_alt_storage;

  // Host-side boolean whether or not an odd number of sorting passes left the 
  // results in alternate storage.  If so, the d_keys (and d_values) pointers 
  // will have been swapped with the d_alt_keys (and d_alt_values) pointers in order to 
  // point to the final results.
  bool using_alternate_storage;
	
  // Constructor
  RadixSortStorage(K* keys = NULL, V* values = NULL) 
  { 
    d_keys = keys; 
    d_values = values; 
    d_alt_keys = NULL; 
    d_alt_values = NULL; 
    d_spine = NULL;
    d_from_alt_storage = NULL;
    d_useful = NULL;

    using_alternate_storage = false;
  }

  RadixSortStorage(uint k, uint num_elements,uint need,K* keys = NULL, V* values = NULL) 
  { 
    h_useful[0] = 50;
    h_useful[1] = num_elements - k;
    h_useful[2] = 0;
    h_useful[3] = num_elements - 1;
    h_useful[4] = num_elements;
    h_useful[5] = 0;
    h_useful[6] = num_elements - 1;          
    h_useful[7] = k;
    needToPre = need;
    d_useful = NULL;
    d_storeCounts = NULL;

    d_keys = keys; 
    d_values = values; 
    d_alt_keys = NULL; 
    d_alt_values = NULL; 
    d_spine = NULL;
    d_from_alt_storage = NULL;
                
    using_alternate_storage = true;
  }
  // Clean up non-results storage (may include freeing original storage if 
  // primary pointers were swizzled as per using_alternate_storage) 
  cudaError_t CleanupTempStorage() 
  {
    if (d_alt_keys) cudaFree(d_alt_keys);
    if (d_alt_values) cudaFree(d_alt_values);
    if (d_spine) cudaFree(d_spine);
    if (d_from_alt_storage) cudaFree(d_from_alt_storage);
		
    return cudaSuccess;
  }
};



/******************************************************************************
 * Base class for sorting enactors
 ******************************************************************************/


/**
 * Base class for SRTS radix sorting enactors.
 */
template <typename K, typename V>
class BaseRadixSortingEnactor 
{
public:
	
	// Unsigned integer type suitable for radix sorting of keys
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

protected:

	//
	// Information about our problem configuration
	//
	
	bool				_keys_only;
	unsigned int 		_num_elements;
	int 				_cycle_elements;
	int 				_spine_elements;
	int 				_grid_size;
	CtaDecomposition 	_work_decomposition;
	int 				_passes;
	bool 				_swizzle_pointers_for_odd_passes;

	// Information about our target device
	cudaDeviceProp 		_device_props;
	int 				_device_sm_version;
	
	// Information about our kernel assembly
	int 				_kernel_ptx_version;
	cudaFuncAttributes 	_spine_scan_kernel_attrs;
	
protected:
	
	/**
	 * Constructor.
	 */
	BaseRadixSortingEnactor(int passes, int radix_bits, unsigned int num_elements, int max_grid_size, bool swizzle_pointers_for_odd_passes = true); 
	
	/**
	 * Heuristic for determining the number of CTAs to launch.
	 *   
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  A value of 0 indicates 
	 * 		that the default value should be used.
	 * 
	 * @return The actual number of CTAs that should be launched
	 */
	int GridSize(int max_grid_size);

	/**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
	uint DigitPlacePass(const RadixSortStorage<ConvertedKeyType, V> &converted_storage); 
	
	/**
	 * Enacts a sorting operation by performing the the appropriate 
	 * digit-place passes.  To be overloaded by specialized subclasses.
	 */
	virtual uint EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses0(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses1(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses2(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses3(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses4(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses5(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses6(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses7(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses8(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses9(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses10(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses11(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses12(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses13(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses14(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	virtual uint EnactDigitPlacePasses15(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	
public:
	
	/**
	 * Returns the length (in unsigned ints) of the device vector needed for  
	 * temporary storage of the reduction spine.  Useful if pre-allocating 
	 * your own device storage (as opposed to letting EnactSort() allocate it
	 * for you).
	 */
	int SpineElements() { return _spine_elements; }

	/**
	 * Returns whether or not the problem will fit on the device.
	 */
	bool CanFit();

	/**
	 * Enacts a radix sorting operation on the specified device data.
	 * 
	 * IMPORTANT NOTES: The device storage backing the specified input vectors of 
	 * keys (and data) will be modified.  (I.e., treat this as an in-place sort.)  
	 * 
	 * Additionally, the pointers in the problem_storage structure may be updated 
	 * (a) depending upon the number of digit-place sorting passes needed, and (b) 
	 * whether or not the caller has already allocated temporary storage.  
	 * 
	 * The sorted results will always be referenced by problem_storage.d_keys (and 
	 * problem_storage.d_values).  However, for an odd number of sorting passes (uncommon)
	 * these results will actually be backed by the storage initially allocated for 
	 * by problem_storage.d_alt_keys (and problem_storage.d_alt_values).  If so, 
	 * problem_storage.d_alt_keys and problem_storage.d_alt_keys will be updated to 
	 * reference the original problem_storage.d_keys and problem_storage.d_values in order 
	 * to facilitate cleanup.  
	 * 
	 * This means it is important to avoid keeping stale copies of device pointers 
	 * to keys/data; you will want to re-reference the pointers in problem_storage.
	 * 
	 * @param[in/out] 	problem_storage 
	 * 		Device vectors of keys and values to sort, and ancillary storage 
	 * 		needed by the sorting kernels. See the IMPORTANT NOTES above. 
	 * 
	 * 		The problem_storage.[alternate_keys|alternate_values|d_spine] fields are 
	 * 		temporary storage needed by the sorting kernels.  To facilitate 
	 * 		speed, callers are welcome to re-use this storage for same-sized 
	 * 		(or smaller) sortign problems. If NULL, these storage vectors will be 
	 *      allocated by this routine (and must be subsequently cuda-freed by 
	 *      the caller).
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	uint EnactSort(RadixSortStorage<K, V> &problem_storage, uint pass);	

    /*
     * Destructor
     */
    virtual ~BaseRadixSortingEnactor() {}
};



template <typename K, typename V>
BaseRadixSortingEnactor<K, V>::BaseRadixSortingEnactor(
	int passes, 
	int max_radix_bits, 
	unsigned int num_elements, 
	int max_grid_size,
	bool swizzle_pointers_for_odd_passes) 
{
	//
	// Get current device properties 
	//

	int current_device;
	cudaGetDevice(&current_device);
	cudaGetDeviceProperties(&_device_props, current_device);
	_device_sm_version = _device_props.major * 100 + _device_props.minor * 10;

	
	//
	// Get SM version of compiled kernel assembly
	//
	cudaFuncGetAttributes(&_spine_scan_kernel_attrs, SrtsScanSpine<void>);
	_kernel_ptx_version = _spine_scan_kernel_attrs.ptxVersion * 10;
	

	//
	// Determine number of CTAs to launch, shared memory, cycle elements, etc.
	//

	_passes								= passes;
	_num_elements 						= num_elements;
	_keys_only 							= IsKeysOnly<V>();
	_cycle_elements 					= B40C_RADIXSORT_CYCLE_ELEMENTS(_kernel_ptx_version , ConvertedKeyType, V);
	_grid_size 							= GridSize(max_grid_size);
	_swizzle_pointers_for_odd_passes	= swizzle_pointers_for_odd_passes;
	
	int total_cycles 			= _num_elements / _cycle_elements;
	int cycles_per_block 		= total_cycles / _grid_size;						
	int extra_cycles 			= total_cycles - (cycles_per_block * _grid_size);

	CtaDecomposition work_decomposition = {
		extra_cycles,										// num_big_blocks
		(cycles_per_block + 1) * _cycle_elements,			// big_block_elements
		cycles_per_block * _cycle_elements,					// normal_block_elements
		_num_elements - (total_cycles * _cycle_elements),	// extra_elements_last_block
		_num_elements};										// num_elements
	
	_work_decomposition = work_decomposition;
	
	int spine_cycles = ((_grid_size * (1 << max_radix_bits)) + B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS - 1) / B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
	_spine_elements = spine_cycles * B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
}



template <typename K, typename V>
int BaseRadixSortingEnactor<K, V>::GridSize(int max_grid_size)
{
	const int SINGLE_CTA_CUTOFF = 0;		// right now zero; we have no single-cta sorting

	// find maximum number of threadblocks if "use-default"
	if (max_grid_size == 0) {

		if (_num_elements <= static_cast<unsigned int>(SINGLE_CTA_CUTOFF)) {

			// The problem size is too small to warrant a two-level reduction: 
			// use only one stream-processor
			max_grid_size = 1;

		} else {

			if (_device_sm_version <= 120) {
				
				// G80/G90
				max_grid_size = _device_props.multiProcessorCount * 4;
				
			} else if (_device_sm_version < 200) {
				
				// GT200 (has some kind of TLB or icache drama)
				int orig_max_grid_size = _device_props.multiProcessorCount * B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(_kernel_ptx_version);
				if (_keys_only) { 
					orig_max_grid_size *= (_num_elements + (1024 * 1024 * 96) - 1) / (1024 * 1024 * 96);
				} else {
					orig_max_grid_size *= (_num_elements + (1024 * 1024 * 64) - 1) / (1024 * 1024 * 64);
				}
				max_grid_size = orig_max_grid_size;

				if (_num_elements / _cycle_elements > static_cast<unsigned int>(max_grid_size)) {
	
					double multiplier1 = 4.0;
					double multiplier2 = 16.0;

					double delta1 = 0.068;
					double delta2 = 0.127;	
	
					int dividend = (_num_elements + _cycle_elements - 1) / _cycle_elements;
	
					while(true) {
	
						double quotient = ((double) dividend) / (multiplier1 * max_grid_size);
						quotient -= (int) quotient;

						if ((quotient > delta1) && (quotient < 1 - delta1)) {

							quotient = ((double) dividend) / (multiplier2 * max_grid_size / 3.0);
							quotient -= (int) quotient;

							if ((quotient > delta2) && (quotient < 1 - delta2)) {
								break;
							}
						}
						
						if (max_grid_size == orig_max_grid_size - 2) {
							max_grid_size = orig_max_grid_size - 30;
						} else {
							max_grid_size -= 1;
						}
					}
				}
			} else {
				
				// GF100
				max_grid_size = 418;
			}
		}
	}

	// Calculate the actual number of threadblocks to launch.  Initially
	// assume that each threadblock will do only one cycle_elements worth 
	// of work, but then clamp it by the "max" restriction derived above
	// in order to accomodate the "single-sp" and "saturated" cases.

	int grid_size = _num_elements / _cycle_elements;
	if (grid_size == 0) {
		grid_size = 1;
	}
	if (grid_size > max_grid_size) {
		grid_size = max_grid_size;
	} 

	return grid_size;
}

__global__ void determineRadixOfInterest(int *d_spine, uint *d_useful, uint blocks)
  {
    __shared__ uint lOffsets1[17];
    __shared__ uint lOffsets2[17];
    uint oldK = d_useful[4] - d_useful[1];
    uint idx = threadIdx.x;
    if(idx == 16){
      lOffsets1[15] = d_useful[3] + 1;
      lOffsets1[16] = d_useful[3] + 1;
      lOffsets2[16] = d_useful[3] + 1;
      lOffsets2[15] = d_useful[3] + 1;
    }
  
    if(idx < 15){
      lOffsets1[idx] = d_spine[(idx +1) *blocks] + d_useful[2];
      lOffsets2[idx] = lOffsets1[idx];
    }
    __syncthreads();
  
    if(lOffsets2[0] > d_useful[1]){
      if(idx == 0){
        //save old start and stop
        d_useful[5] = d_useful[2];
        d_useful[6] = d_useful[3];
        //radix of interest is the threadIdx.x 
        d_useful[0] = 0;
        d_useful[2] = 0;
        d_useful[3] = lOffsets2[0] - 1;
        d_useful[7] = oldK - (d_useful[4] -(d_useful[3] + 1));
        d_useful[4] = d_useful[3] - d_useful[2] + 1;
        d_useful[1] = d_useful[4] - d_useful[7];
      }
    }
    else if(idx == 0){
      for(; idx< 16; idx++){
      if(lOffsets2[idx + 1] > d_useful[1] && lOffsets1[idx] <= d_useful[1]){
        //save old start and stop
        d_useful[5] = d_useful[2];
        d_useful[6] = d_useful[3];
        //radix of interest is the threadIdx.x t
        d_useful[0] = (uint)idx + 1;
        d_useful[2] = lOffsets1[idx];//the start point
        d_useful[3] = lOffsets2[idx+1] - 1;//the stop point
        d_useful[7] = oldK - (d_useful[4] -(d_useful[3] + 1));//new value of K 
        d_useful[4] = d_useful[3] - d_useful[2] + 1;//the new size
        d_useful[1] = d_useful[4] - d_useful[7];// index of kth
        break;
      }
    }
  }
  }


template <typename K, typename V>
bool BaseRadixSortingEnactor<K, V>::
CanFit() 
{
	long long bytes = (_num_elements * sizeof(K) * 2) + (_spine_elements * sizeof(int));
	if (!_keys_only) bytes += _num_elements * sizeof(V) * 2;

	if (_device_props.totalGlobalMem < 1024 * 1024 * 513) {
		return (bytes < ((double) _device_props.totalGlobalMem) * 0.81); 	// allow up to 81% capacity for 512MB   
	}
	
	return (bytes < ((double) _device_props.totalGlobalMem) * 0.89); 	// allow up to 90% capacity 
}



template <typename K, typename V>
template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
uint BaseRadixSortingEnactor<K, V>::
DigitPlacePass(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
{
  uint use[8];
  //store the incoming size, use for comparision to new size ater
  uint inSize = converted_storage.h_useful[4];
  //the following is from RadixSort, it sets up variables used for problem set up
  int threads = B40C_RADIXSORT_THREADS;
  int dynamic_smem;
  cudaFuncAttributes reduce_kernel_attrs, scan_scatter_attrs;
  cudaFuncGetAttributes(&reduce_kernel_attrs, RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor>);
  cudaFuncGetAttributes(&scan_scatter_attrs, ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor>);
	
  //
  // Counting Reduction
  //

  // Run tesla flush kernel if we have two or more threadblocks for each of the SMs
  if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
    FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }

  // GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
  dynamic_smem = (_kernel_ptx_version >= 130) ? scan_scatter_attrs.sharedSizeBytes - reduce_kernel_attrs.sharedSizeBytes : 0;

  RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor>
    <<<_grid_size, threads, dynamic_smem>>>(
                                            converted_storage.d_from_alt_storage,
                                            converted_storage.d_spine, 
                                            converted_storage.d_keys,
                                            converted_storage.d_alt_keys,
                                            _work_decomposition);
  thrust::detail::device::cuda::synchronize_if_enabled("RakingReduction");


  //
  // Spine
  //
	
  // GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
  dynamic_smem = (_kernel_ptx_version >= 130) ? scan_scatter_attrs.sharedSizeBytes - _spine_scan_kernel_attrs.sharedSizeBytes : 0;

  SrtsScanSpine<void><<<_grid_size, B40C_RADIXSORT_SPINE_THREADS, dynamic_smem>>>(
                                                                                  converted_storage.d_spine,
                                                                                  converted_storage.d_spine,
                                                                                  _spine_elements);
  thrust::detail::device::cuda::synchronize_if_enabled("SrtsScanSpine");
  //at this point we can determine the number of elements in the list that have each radix. Actually the number that have 
  //a radix less than or equal that radix. We use hese counts to determine the radix of the kth largest element. 
  determineRadixOfInterest<<<1,17, 40 * sizeof(unsigned int)>>>(converted_storage.d_spine,converted_storage.d_useful, _grid_size);
  thrust::detail::device::cuda::synchronize_if_enabled("determineRadixOfInterest");
	
  //
  // Scanning Scatter
  //


  // Run tesla flush kernel if we have two or more threadblocks for each of the SMs
  if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
    printf("FLUSHING\n");
    FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }
  //copy d_useful, which contains the updated information about the problem to use, which is on the host
  cudaMemcpy(&use, converted_storage.d_useful, 8*sizeof(uint), cudaMemcpyDeviceToHost);
  //if the inSize, is the same as the new problem  size, then we know that all elements of this list have the
  //same radix so we do not need to scatter the list, which is very time consuming.
  //however we need comunicate this to other functions, so they know where the list is, we do this by returning 1
 if(inSize == use[4]){
   return 1;
 }

  //otherwise we do need to scatter the list, so we do so, and then return 0.
  else{

    ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor>
      <<<_grid_size, threads, 0>>>(
                                   converted_storage.d_from_alt_storage,
                                   converted_storage.d_spine,
                                   converted_storage.d_keys,
                                   converted_storage.d_alt_keys,
                                   converted_storage.d_values,
                                   converted_storage.d_alt_values,
                                   _work_decomposition);
    thrust::detail::device::cuda::synchronize_if_enabled("ScanScatterDigits");

    return 0;
  }

}



template <typename K, typename V>
uint BaseRadixSortingEnactor<K, V>::
  EnactSort(RadixSortStorage<K, V> &problem_storage,uint pass) 
{

	//
	// Allocate device memory for temporary storage (if necessary)
	//
	if (problem_storage.d_alt_keys == NULL) {
		cudaMalloc((void**) &problem_storage.d_alt_keys, _num_elements * sizeof(K));
	}
	if (!_keys_only && (problem_storage.d_alt_values == NULL)) {
		cudaMalloc((void**) &problem_storage.d_alt_values, _num_elements * sizeof(V));
	}
	if (problem_storage.d_spine == NULL) {
		cudaMalloc((void**) &problem_storage.d_spine, _spine_elements * sizeof(int));
	}
	if (problem_storage.d_from_alt_storage == NULL) {
		cudaMalloc((void**) &problem_storage.d_from_alt_storage, 2 * sizeof(bool));
	}

        if(problem_storage.d_useful == NULL){
          cudaMalloc((void**) &problem_storage.d_useful, 8 * sizeof(uint));

        }
	// Determine suitable type of unsigned byte storage to use for keys 
        typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;
        cudaMemcpy(problem_storage.d_useful, &problem_storage.h_useful, 8 *sizeof(uint), cudaMemcpyHostToDevice);
   

	// Copy storage pointers to an appropriately typed stucture 
        //NOTE: SOME OF THIS COPYING IS LIKELY REDUNDANT
	RadixSortStorage<ConvertedKeyType, V> converted_storage;
	memcpy(&converted_storage, &problem_storage, sizeof(RadixSortStorage<K, V>));
	cudaMemcpy(converted_storage.d_useful, problem_storage.d_useful, 8 *sizeof(uint), cudaMemcpyDeviceToDevice);
	// 
	// Enact the sorting operation
	//
        //
        //initialize retval, 55 is arbitrary, it is just not 1 or 0, also it is high enough that
        // it is doubtful that there are ever going to be that many different states to comunicate. 
        uint retval=55;
        if(pass ==0){
          retval = EnactDigitPlacePasses0(converted_storage);
        }
        else if(pass ==1){
          retval =    EnactDigitPlacePasses1(converted_storage);
        }
        else if(pass ==2){
          retval =   EnactDigitPlacePasses2(converted_storage);
        }
        else if(pass ==3){
          retval =  EnactDigitPlacePasses3(converted_storage);
        }
        else if(pass ==4){
          retval = EnactDigitPlacePasses4(converted_storage);
        }
        else if(pass ==5){
          retval =  EnactDigitPlacePasses5(converted_storage);
        }
        else if(pass ==6){
          retval =  EnactDigitPlacePasses6(converted_storage);
        }
        else if(pass ==7){
          retval =    EnactDigitPlacePasses7(converted_storage);
	}	
        else if(pass ==8){
          retval =    EnactDigitPlacePasses8(converted_storage);
        }
        else if(pass ==9){
          retval =   EnactDigitPlacePasses9(converted_storage);
        }
        else if(pass ==10){
          retval =  EnactDigitPlacePasses10(converted_storage);
        }
        else if(pass ==11){
          retval = EnactDigitPlacePasses11(converted_storage);
        }
        else if(pass ==12){
          retval =  EnactDigitPlacePasses12(converted_storage);
        }
        else if(pass ==13){
          retval =  EnactDigitPlacePasses13(converted_storage);
        }
        else if(pass ==14){
          retval =    EnactDigitPlacePasses14(converted_storage);
	}
        else if(pass ==15){
          retval =    EnactDigitPlacePasses15(converted_storage);
	}
	//
	// Swizzle pointers if we left our sorted output in temp storage 
	//
        if(! retval){
          thrust::swap<K*>(problem_storage.d_keys, problem_storage.d_alt_keys);
          }
    
        cudaMemcpy(problem_storage.h_useful,converted_storage.d_useful, 8 *sizeof(uint), cudaMemcpyDeviceToHost);

	return retval;
}





/******************************************************************************
 * Sorting enactor classes
 ******************************************************************************/

/**
 * Generic sorting enactor class.  Simply create an instance of this class
 * with your key-type K (and optionally value-type V if sorting with satellite 
 * values).
 * 
 * Template specialization provides the appropriate enactor instance to handle 
 * the specified data types. 
 * 
 * @template-param K
 * 		Type of keys to be sorted
 *
 * @template-param V
 * 		Type of values to be sorted.
 *
 * @template-param ConvertedKeyType
 * 		Leave as default to effect necessary enactor specialization.
 */
template <typename K, typename V = KeysOnlyType, typename ConvertedKeyType = typename KeyConversion<K>::UnsignedBits>
class RadixSortingEnactor;



/**
 * Sorting enactor that is specialized for for 32-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned int> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;
        uint EnactDigitPlacePasses0(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 28,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 28, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses1(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 24,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 24, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses2(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 20,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 20, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses3(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 16,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 16, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses4(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 12,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 12, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses5(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 8,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 8, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses6(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 4,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 4, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses7(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 0, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses8(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses9(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses10(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses11(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses12(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses13(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses14(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}
	uint EnactDigitPlacePasses15(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}

	uint EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(8, 4, num_elements, max_grid_size) {}

};

template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned long long> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;
  
       uint EnactDigitPlacePasses0(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){

       return   Base::template DigitPlacePass<0, 4, 60,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 60, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
       uint EnactDigitPlacePasses1(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){

          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 56,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 56, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses2(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){

          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 52,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 52, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses3(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){

          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 48,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 48, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses4(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 44,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 44, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses5(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 40,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 40, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses6(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 36,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 36, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
        uint EnactDigitPlacePasses7(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4,32 ,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 32, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
       
        uint EnactDigitPlacePasses8(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 28,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 28, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses9(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){

       return   Base::template DigitPlacePass<0, 4, 24,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 24, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses10(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 20,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 20, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses11(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 16,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 16, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses12(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 12,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 12, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses13(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 8,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 8, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }

	uint EnactDigitPlacePasses14(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 4,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 4, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses15(const RadixSortStorage<ConvertedKeyType, V> &converted_storage){
          if(converted_storage.needToPre){
       return   Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
          else{
            return   Base::template DigitPlacePass<0, 4, 0, NopFunctor<ConvertedKeyType> ,      NopFunctor<ConvertedKeyType> >(converted_storage);
          }
        }
	uint EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{	
          return 0;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(8, 4, num_elements, max_grid_size) {}

};


} //end namespace RadixSelect
