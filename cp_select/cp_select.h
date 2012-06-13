

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


//#define MASHEPS 1e-12

namespace cp_select
{
  // Call this to get rid of the overhead for the first cuda call before measuring anything.
  void warmUp();
  void cleanUp();

/* The two functions below are the interfaces to cp_select methods, which can be can be called from a
   C++ program. They copy the array D from host to device and call the appropriate internal methods
   of selection. 
   The parameters are: n the size of the array (it is redundant, as it can be calculated form D)
   CP iter - max number of cutting plane iterations, usually 7 is sufficient.
   method=1 cp_select,  method=2 - use thrust radix sort for selection.
   k is the k-th order statistic, k starts with 1 ( not 0!)
   if omitted or 0, then the median which is (n+1)/2-order statistic, is calculated
*/
	float median_min_float( thrust::host_vector<float> &D,      int n, int CPiters,  int method, int k=0);
	double  median_min_double( thrust::host_vector<double> &D,  int n, int CPiters,  int method, int k=0);	
	
	
/*========================================================================*/
/* The methods below can be called from .cu, but seem not to be properly used by the linker
 when called from a .c++ program.
*/
	
	template<typename InputIterator, typename Tbase, typename Tsum >
	int mincp( Tbase a, Tbase b, Tbase sum, int maxint, Tbase *x , Tbase* L, Tbase* R, Tsum * left, Tsum * right,
		InputIterator first, InputIterator last, int n, int k);

	template<typename InputIterator>
	typename thrust::iterator_traits<InputIterator>::value_type
	median_min(InputIterator first, InputIterator last,   int CPiters,  int method, int k=0);

	template<typename InputIterator>
	typename thrust::iterator_traits<InputIterator>::value_type
	median_min_host(InputIterator first, InputIterator last,   int CPiters,  int method, int k=0);


} // namespace


