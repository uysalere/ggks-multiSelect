#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

namespace BucketSelect{

  template <typename T>
  void bucketMultiselectWrapper (T * d_vector, int length, uint * kVals, uint kCount, T * ouputs, int blocks, int threads) {
    return;
  }

}
