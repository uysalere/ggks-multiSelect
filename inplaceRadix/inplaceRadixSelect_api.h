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
namespace InplaceRadix {

  struct compare_radix {
  compare_radix(uint radix, uint bit) : radix(radix) {}
    __device__ void operator()(uint &y) {
      if(y != 0xFFFFFFFF && y !=0x00000000){
        uint temp = (y >> bit) & 0x0000000F;
        if(temp < radix){
          y = 0x00000000;
        }
        else if( temp > radix){
          y = 0xFFFFFFFF;
        }
      }
    }
    __device__ void operator()(unsigned long long &y) {
      if(y != 0xFFFFFFFFFFFFFFFF && y !=0x0000000000000000){
        uint temp = (y >> bit) & 0x000000000000000F;
        if(temp < radix){
          y = 0x0000000000000000;
        }
        else if( temp > radix){
          y = 0xFFFFFFFFFFFFFFFF;
        }
      }
    }
    uint radix, bit;
  };
 
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
	uint *radixes;
        uint *d_radix;
	uint *indexOfK;
	uint size;
	// Flip-flopping temporary device storage denoting which digit place 
	// pass should read from which input source (i.e., false if reading from 
	// keys, true if reading from alternate_keys
	bool *d_from_alt_storage;
	bool *h_from_alt_storage;
	// Host-side boolean whether or not an odd number of sorting passes left the 
	// results in alternate storage.  If so, the d_keys (and d_values) pointers 
	// will have been swapped with the d_alt_keys (and d_alt_values) pointers in order to 
	// point to the final results.
	bool using_alternate_storage;
	uint bitResult;
	// Constructor
	RadixSortStorage(K* keys = NULL, V* values = NULL) 
	{ 
		d_keys = keys; 
		d_values = values; 
		d_alt_keys = NULL; 
		d_alt_values = NULL; 
		d_spine = NULL;
		d_from_alt_storage = NULL;
		
		using_alternate_storage = false;
	}
	RadixSortStorage(uint insize, uint idk, K* keys = NULL, V* values = NULL) 
	{ 
		bitResult = 0;
		indexOfK =(uint *) malloc(1 * sizeof(uint));
                indexOfK[0] = idk;
		size = insize;
		radixes = (uint *) malloc(sizeof(K)*2 * sizeof(uint));
		d_keys = keys; 
		d_values = values; 
		d_alt_keys = NULL; 
		d_alt_values = NULL; 
		d_spine = NULL;
		d_from_alt_storage = NULL;
		
		using_alternate_storage = false;
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
	cudaError_t DigitPlacePass(const RadixSortStorage<uint, V> &converted_storage); 

	template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
	cudaError_t DigitPlacePass(const RadixSortStorage<unsigned long long, V> &converted_storage); 	
	/**
	 * Enacts a sorting operation by performing the the appropriate 
	 * digit-place passes.  To be overloaded by specialized subclasses.
	 */
	virtual cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	
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


	cudaError_t EnactSort(RadixSortStorage<K, V> &problem_storage);	

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

 __global__ void determineRadixOfInterest(int *d_spine, uint *d_radix,uint indexOfK,uint size, uint blocks)
 {
   __shared__ uint lOffsets1[17];
   __shared__ uint lOffsets2[17];
   uint idx = threadIdx.x;
   if(idx == 16){
     lOffsets1[15] = size + 1;
     lOffsets1[16] = size + 1;
     lOffsets2[16] = size + 1;
     lOffsets2[15] = size + 1;
   }
  
   if(idx < 15){
     lOffsets1[idx] = d_spine[(idx +1) *blocks];
     lOffsets2[idx] = lOffsets1[idx];
   }
   __syncthreads();
  
	
   if(lOffsets2[0] > indexOfK){
     if(idx == 0){
       d_radix[0] = idx;
     }
   }

   else if(idx == 0){
     for(; idx< 16; idx++){
       if(lOffsets2[idx + 1] > indexOfK && lOffsets1[idx] <= indexOfK){
         d_radix[0] = idx+1;
         break;
       }
     }
   }
 }

template <typename K, typename V>
template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
cudaError_t BaseRadixSortingEnactor<K, V>::
  DigitPlacePass(const RadixSortStorage<uint, V> &converted_storage){

  compare_radix compareFunctor(0,PASS);
  int threads = B40C_RADIXSORT_THREADS;
  int dynamic_smem;
  uint *radix;
  radix = converted_storage.d_radix;
  cudaMalloc(&radix, sizeof(uint));

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

  RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor> <<<_grid_size, threads, dynamic_smem>>>(
                                                                                                                         converted_storage.d_from_alt_storage,
                                                                                                                         converted_storage.d_spine,
                                                                                                                         converted_storage.d_keys,
                                                                                                                         converted_storage.d_keys,
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

	
  //
  // Scanning Scatter
  //

  determineRadixOfInterest<<<1,17, 40 * sizeof(unsigned int)>>>(converted_storage.d_spine,radix, converted_storage.indexOfK[0],converted_storage.size, _grid_size);
  thrust::detail::device::cuda::synchronize_if_enabled("determineRadixOfInterest");
  // Run tesla flush kernel if we have two or more threadblocks for each of the SMs
  if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
    FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }
 
  cudaMemcpy(converted_storage.radixes + PASS, radix, sizeof(uint), cudaMemcpyDeviceToHost);
  if(PASS != 7){
    compareFunctor.radix = converted_storage.radixes[PASS];
    compareFunctor.bit = BIT;
    thrust::device_ptr<unsigned int> dev_ptr(converted_storage.d_keys);

    thrust::for_each(dev_ptr, dev_ptr + converted_storage.size, compareFunctor);

    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }
  return cudaSuccess;
 }

//THE DOUBLES VERSION
template <typename K, typename V>
template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
cudaError_t BaseRadixSortingEnactor<K, V>::
  DigitPlacePass(const RadixSortStorage<unsigned long long, V> &converted_storage){
  compare_radix compareFunctor(0,PASS);
  int threads = B40C_RADIXSORT_THREADS;
  int dynamic_smem;
  uint *radix;
  //uint *counts;
  cudaMalloc(&radix, sizeof(uint));

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

  RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor> <<<_grid_size, threads, dynamic_smem>>>(
                                                                                                                         converted_storage.d_from_alt_storage,
                                                                                                                         converted_storage.d_spine,
                                                                                                                         converted_storage.d_keys,
                                                                                                                         converted_storage.d_keys,
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

	
  //
  // Scanning Scatter
  //

  determineRadixOfInterest<<<1,17, 40 * sizeof(unsigned int)>>>(converted_storage.d_spine,radix, converted_storage.indexOfK[0],converted_storage.size, _grid_size);
  thrust::detail::device::cuda::synchronize_if_enabled("determineRadixOfInterest");
  // Run tesla flush kernel if we have two or more threadblocks for each of the SMs
  if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
    FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }
 
  cudaMemcpy(converted_storage.radixes + PASS, radix, sizeof(uint), cudaMemcpyDeviceToHost);
  if(PASS != 15){
    compareFunctor.radix = converted_storage.radixes[PASS];
    compareFunctor.bit = BIT;
    thrust::device_ptr<unsigned long long> dev_ptr(converted_storage.d_keys);

    thrust::for_each(dev_ptr, dev_ptr + converted_storage.size, compareFunctor);

    thrust::detail::device::cuda::synchronize_if_enabled("FlushKernel");
  }
  return cudaSuccess;
 }


template <typename K, typename V>
cudaError_t BaseRadixSortingEnactor<K, V>::
EnactSort(RadixSortStorage<K, V> &problem_storage) 
{
	//
	// Allocate device memory for temporary storage (if necessary)
	//

	if (problem_storage.d_spine == NULL) {
		cudaMalloc((void**) &problem_storage.d_spine, _spine_elements * sizeof(int));
	}
	if (problem_storage.d_from_alt_storage == NULL) {
		cudaMalloc((void**) &problem_storage.d_from_alt_storage, 2 * sizeof(bool));
	}

	// Determine suitable type of unsigned byte storage to use for keys 
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;
	
	// Copy storage pointers to an appropriately typed stucture 
	RadixSortStorage<ConvertedKeyType, V> converted_storage;
	memcpy(&converted_storage, &problem_storage, sizeof(RadixSortStorage<K, V>));

	// 
	// Enact the sorting operation
	//
	
	cudaError_t retval = EnactDigitPlacePasses(converted_storage);
	
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

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		Base::template DigitPlacePass<0, 4, 28, NopFunctor<ConvertedKeyType>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1, 4, 24,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<2, 4, 20 , NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<3, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<4, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<5, 4, 8, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<6, 4, 4, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<7, 4, 0, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >    (converted_storage); 
		return cudaSuccess;
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



/**
 * Sorting enactor that is specialized for for 64-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned long long> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
          Base::template DigitPlacePass<0,  4, 60, NopFunctor<ConvertedKeyType>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1,  4, 56,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<2,  4, 52,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<3,  4, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<4,  4, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<5,  4, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<6,  4, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<7,  4, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<8,  4, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<9,  4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<10, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<11, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<12, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<13, 4, 8, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<14, 4, 4, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<15, 4, 0, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >    (converted_storage); 

		return cudaSuccess;
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
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(16, 4, num_elements, max_grid_size) {}

};

}

