/* these functions are based on functions provided in radixsort_key_conversion.h
 * which is provided in thrust */

template<typename T>
struct preProcess{
  __device__ __host__  void operator()(T &converted_key) {}
};
  
template<>
struct preProcess<float>{
  __device__ __host__  void operator()(unsigned int &converted_key) {
    unsigned int mask = (converted_key & 0x80000000) ? 0xffffffff : 0x80000000; 
    converted_key ^= mask;
  }
};
    
template<>
struct preProcess<double>{
  __device__ __host__  void operator()(unsigned long long &converted_key) {
    unsigned long long mask = (converted_key & 0x8000000000000000) ? 0xffffffffffffffff : 0x8000000000000000; 
    converted_key ^= mask;
  }
};


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
