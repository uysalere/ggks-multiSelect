/*The following appears in the readme.txt that comes with the Cp_select library 
 *
 *10 April 2011
 *author Gleb Beliakov (gleb@deakin.edu.au)
 *CP_Select library (Cutting Plane selection algorithm). Freeware.
 *
 *This is the library cp_select Version 1.0 for parallel selection on GPU and 
 *a test program illustrating it use.
 *Comments on the usage of cp_select are in cp_select.h file.

 *Both sources depend on Thrust library, which should be installed first, and of course, on CUDA.

 *compile both files using ./compileme script
 *or any other appropriate method (eg. converting it to a project for Visual C++)

 *If you are interested in timing the selection methods, uncomment the line
//#define DEBUG__CPS 
*in cp_select.cu file.

*nvcc -O2 -c -arch=sm_20 cp_select.cu
*is to compile the library only. The option -arch=sm_20 (or -arch=sm_13) is to  ensure 
*doubles are not downgraded to floats
*/
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h> 
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/sort.h>



//#define DEBUG__CPS 

/*=============================================*/
#ifdef DEBUG__CPS
/* This is just for timing the selection algorithm once the data are on GPU (exclude transfer) .*/


#ifdef _MSC_VER
#include <windows.h>
	LARGE_INTEGER starttime;
void start(){ 	QueryPerformanceCounter(&starttime);}

double stop()
{
	LARGE_INTEGER endtime,freq;
	QueryPerformanceCounter(&endtime);
	QueryPerformanceFrequency(&freq);
	return ((double)(endtime.QuadPart-starttime.QuadPart))/((double)(freq.QuadPart/1000.0));
}

#else // linux

 #include <sys/time.h>
 struct timeval starttime;
 void start() {	gettimeofday(&starttime,0); }

double stop()
{
	struct timeval endtime;
	gettimeofday(&endtime,0);
	return (endtime.tv_sec - starttime.tv_sec)*1000.0 + (endtime.tv_usec - starttime.tv_usec)/1000.0;
}
#endif // windows

#endif // debug
/*=============================================*/


#define MACHEPS_FLOAT 1e-7
#define MACHEPS_DOUBLE 1e-18
#define LARGE_NUMBER_DOUBLE 10e12
#define LARGE_NUMBER_FLOAT 10e12
#define MULT 1e0

namespace cp_select
{

  // Call this to get rid of the overhead for the first cuda call before measuring anything.
  void warmUp()
  {
    int *dData = 0;
    cudaMalloc((void**) &dData, sizeof(dData));
    cudaFree(dData); 
  }

template <typename T>
struct transform_functor : public thrust::unary_function<T,T>
{
	const T A;
	transform_functor (T _a) : A(_a) {}

    __host__ __device__
    T operator()( T& a)
    {    a=log(MULT*(A+a)+1.0);  return 0;}
};


template <typename T>
T transformback(T A, double mult, T transformedcoefficient)
{
	return (exp(A)-1.0)/MULT+transformedcoefficient;
}


template <typename T,typename TS, typename T1>
struct abs_diff : public thrust::unary_function<T,T1>
{
	const T y;
	abs_diff(T _a) : y(_a) {}

    __host__ __device__
    T1 operator()(const T& a)
    {  
		T r=y-a;
		return T1( TS (fabs(r)), r >= 0);
    }
};

template <typename T,typename TS, typename T1>
struct abs_diff_asymetric : public thrust::unary_function<T,T1>
{
	const T y;
	const TS A,B;
	abs_diff_asymetric(T _a, TS _A, TS _B) : y(_a), A(_A), B(_B) {}

    __host__ __device__
    T1 operator()(const T& a)
    {  
		T r=y-a;
		if (r>=0) return T1( TS (B*r), 1);
		return T1( TS (-A*r), 0);
    }
};


template <typename T>
struct plusplus : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    { 	 return T(a.first+b.first,a.second+b.second); }
};


template <typename T>
struct min_largerA : public thrust::binary_function<T,T,T>
{
	const T A;
	min_largerA (T _a) : A(_a) {}

    __host__ __device__
    T operator()(const T& a, const T& b)
    {  
		if(a>A && b>A) return thrust::min(a,b);
		if(a>A) return a;
		if(b>A) return b;
		return 1e200;
    }
};

template <typename T>
struct max_smallerA : public thrust::binary_function<T,T,T>
{
	const T A;
	max_smallerA (T _a) : A(_a) {}

    __host__ __device__
    T operator()(const T& a, const T& b)
    {  
		if(a<=A && b<=A) return thrust::max(a,b);
		if(a<=A) return a;
		if(b<=A) return b;
		return -1e200;
    }
};


  
template <typename T,typename T1>
struct transtuple : public thrust::unary_function<T,T1>
{
    __host__ __device__
    T1 operator()(const T& a)
    {  
       return T1(a,a,a); 
    }
};

template <typename T>
struct maxminsum : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()( T& a,  T& b)
    {  
		return thrust::make_tuple(fmax(thrust::get<0>(a),thrust::get<0>(b)), 
		  fmin(thrust::get<1>(a),thrust::get<1>(b)), 
		      thrust::get<2>(a)+thrust::get<2>(b));
     }
};

template <typename T>
struct is_inbracket
  {
  	const T L,R;
	is_inbracket(T _a, T _b) : L(_a),R(_b) {}

    __host__ __device__
    bool operator()(const  T &x)
    {
		return (x>L && x<R);
    }
  };


   
template<typename InputIterator, typename Tbase, typename Tsum>
	void ComputeMaxMinSum(InputIterator first, InputIterator last, Tbase & maxA, Tbase & minA, Tsum & sum)
	{
	  thrust::tuple<Tbase,Tbase,Tsum> init(-10e100,10e100,0.0);
	  maxminsum<thrust::tuple<Tbase,Tbase,Tsum> > binaryop;
	  transtuple<Tbase, thrust::tuple<Tbase,Tbase,Tsum> > unary;
	  thrust::tuple<Tbase,Tbase,Tsum> result = thrust::transform_reduce(first, last, unary,  init,  binaryop);

	  minA=thrust::get<1>(result);
	  maxA=thrust::get<0>(result);
	  sum=thrust::get<2>(result);	  
	}	
	
	template<typename InputIterator, typename Tsum>
	void SumAll(InputIterator first, InputIterator last, Tsum &s)
	{
		typedef typename thrust::iterator_traits<InputIterator>::value_type  Tbase;
		s= thrust::reduce(first, last,0.0, thrust::plus<Tsum>());
	}
	
	
	template<typename InputIterator, typename Tbase>
	Tbase ComputeMinLargerA(InputIterator first, InputIterator last, Tbase a)
	{
		 min_largerA<Tbase>    binary_op_min(a);
		 Tbase init=10e120;
		 return	thrust::reduce(first,last,init,binary_op_min);
	}

	template<typename InputIterator, typename Tbase>
	Tbase ComputeMaxSmallerA(InputIterator first, InputIterator last,Tbase a)
	{
		 max_smallerA<Tbase>    binary_op_max(a);
		 Tbase init=-10e120;
		 return 	thrust::reduce(first,last,init,binary_op_max);
	}	
	
	
	template<typename InputIterator, typename Tbase>
	void transform(InputIterator first, InputIterator last, Tbase min_x)
	{
		Tbase transformedcoefficient=min_x;
		transform_functor<Tbase> trans(-transformedcoefficient);
		thrust::for_each(first,last,trans);
	}	
	

								
	template<typename InputIterator, typename Tbase>
	void ComputeBySorting(InputIterator first, InputIterator last, Tbase & A, Tbase  L, Tbase  R, 
				int  LD, int  RD, int idx)
	{
		// here we gather the data and sort
		is_inbracket<Tbase> pred(L, R);
		
		int SIZE=last-first;
		int below=LD;
		int above=SIZE- RD;
		int len = SIZE-(below+above);
#ifdef DEBUG__CPS
		printf("compute by sorting length %d %d %d\n",len,LD,RD);		
#endif
		if(len<2) {  A=L; return; }
		
		thrust::device_vector<Tbase> d_b((len+100)*2); // just in case overestimate the length of pivot array
		InputIterator endme= thrust::copy_if(first,last,d_b.begin(),pred);
		
#ifdef DEBUG__CPS
	//	printf("actual length %d \n",endme-d_b.begin());		
#endif
		if(endme-d_b.begin()>=2)
			thrust::sort(d_b.begin(),endme);
		int index=idx-below;
		if(index>0)
			A=d_b[index]; else A=d_b[0];
		
	}
	
	
/* Calculaiton of the objective in the CP method, and its subgradient. */
	
	template<typename InputIterator,  typename Tsum>
	Tsum CallObjective(InputIterator first, InputIterator last, Tsum t, Tsum & df, int& below)
	{
		typedef typename thrust::iterator_traits<InputIterator>::value_type  Tbase;
		typedef typename thrust::pair<Tsum , unsigned int> Tpair;
		
		abs_diff<Tbase ,Tsum, Tpair>   unary_op(t);
		plusplus<Tpair>        	       binary_op;

		int SIZE=last-first;
		Tpair  result, initpair(0,0);
		result = thrust::transform_reduce(first, last, unary_op,  initpair,  binary_op);	  
		df=2.0*result.second-SIZE;	
		below=result.second;

	    return result.first;
    }	
    
	template<typename InputIterator,  typename Tsum>
	Tsum CallObjective_A(InputIterator first, InputIterator last, Tsum t, Tsum & df, int k, int n, int& below)
	{
		typedef typename thrust::iterator_traits<InputIterator>::value_type  Tbase;
		typedef typename thrust::pair<Tsum , unsigned int> Tpair;
		
		Tsum A = (2.0*k-1)/n;
		Tsum B=2-A;
		abs_diff_asymetric<Tbase ,Tsum, Tpair>   unary_op_a(t, A,B);
		plusplus<Tpair>        	       binary_op;

		int SIZE=last-first;
		Tpair  result, initpair(0,0);
		result = thrust::transform_reduce(first, last, unary_op_a,  initpair,  binary_op);	  
		df=2.0*result.second-SIZE*A;	
		below=result.second;
	    return result.first;
    }	
    


	double solve4x(double a, double b, double fa, double fb, double fa1, double fb1)
	{
		return (fb-fa+a*fa1-b*fb1)/(fa1-fb1);
	}	
	
/* This is the customised Kelley's cutting plane algorithm. It makes a few iterations of the CP algorithm and 
   returns an approximation to the median (or order statistic), the pivot interval containing it, and the number of
   elements of the array smaller or equal to the left and right boundaries.
   The array x is passed through the iterators first and last. a and b are the initial bounds on the median, and sum
   is the sum of all elements of x, precalculated together with a and b in one reduction.
*/	
	template<typename InputIterator, typename Tbase, typename Tsum >
	int mincp( Tbase a, Tbase b, Tsum sum, int maxint, Tbase *x , Tbase* L, Tbase* R, int * belowleft, int * belowright,
		InputIterator first, InputIterator last, int n, int k)
	{
		int i,below;
		double fa=0, fb,fa1,fb1, t=0.0, ft=0.0, ft1=0.0;
		fa=sum;
		Tbase masheps;
		if(sizeof(masheps)<=4)masheps=MACHEPS_FLOAT;
		else masheps=MACHEPS_DOUBLE;

		
		if(fa<=-10e10)  { SumAll(first,last,fa); }

		double A,B;
		A=(2.0*k-1)/n; B=2-A;
		
		int SIZE=n;

		fb = Tsum(SIZE)*Tsum(b) - fa;
		fa = fa-SIZE*Tsum(a);
		
		fa1 =-SIZE+1+B/A;
		fb1 =SIZE-1-A/B;
		
		fa*=A;
		fb*=B; // correction when it's not the median but order statistic
		fa1*=A;
		fb1*=B;
		
#ifdef DEBUG__CPS
	//	printf("CP iteration 0: t, f(t), g(t), a,b  %lf %lf %lf %lf %lf \n",a,fa,fa1,a,b);		
	//	printf("CP iteration 1: t, f(t), g(t), a,b  %lf %lf %lf %lf %lf \n",b,fb,fb1,a,b);		
#endif

		*L=a; *R=b;
		*belowleft=1; *belowright=n-1;

		for(i=0;i<maxint;i++)
		{
			if(fabs(fa1-fb1)<=0.1)  goto stop;

			if( b-a < masheps ) 
			{ *x=a;  *L=a; *R=b; return 4;}
			
			t=solve4x((double)a,(double)b,fa,fb,fa1,fb1);  

			if(fabs(t-a)<masheps  || fabs(t-b)<masheps ) {			
				*x=t; *L=a; *R=b;
				return 4;
			}

			if(k == (SIZE+1)/2)
				ft=CallObjective(first, last, t,ft1,below);
			else
				ft=CallObjective_A(first, last, t,ft1,k,n,below);
#ifdef DEBUG__CPS
	//	printf("CP iteration: t, f(t), g(t), a,b  %lf %lf %lf %lf %lf \n",t,ft,ft1,a,b);		
#endif
			
		if(ft1>=-2.001 && ft1<=1.1)  
		    goto stop;	
		if(ft1<0)
			{
				a=t;
				fa=ft;
				fa1=ft1;
				*L=a;
				*belowleft=below;
			} 
			else if (ft1>0) 
			{
				b=t;
				fb=ft;
				fb1=ft1;
				*R=b;
				*belowright=below;
			} 
			else 
			{
	stop:  // stopping criteria were satisfied
				*x=t;
				if(ft1<0)
					return -1;
				else
					 return ft1;
			}
		}
		*x=t;

		return 3;  // normal exit
	}	
	
			
/* This is the main algorithm. It calls mincp  and then either sorts the pivot interval, or identifies the order statistic 
   directly by reduction, depending on the mincp exit code. 
   The array x (in GPU global memory) is passed through the iterators first and last.
   if method==2 then Thrust's GPU sort algorithm is called (for comparison) 
*/	

template<typename InputIterator>
	typename thrust::iterator_traits<InputIterator>::value_type
	median_min(InputIterator first, InputIterator last,   int CPiters,  int method, int k=0)
	{
#ifdef DEBUG__CPS	
	start();
	double time;
#endif	
	
		typedef typename thrust::iterator_traits<InputIterator>::value_type Tbase;
		typedef  double Tsum;
		int n=last-first;

		// calculates the median by minimisation
		int transformed=0;		
		Tbase  transformedcoefficient=0;
		
		if(k==0) k=(n+1)/2; // it means we need the median
		
		if(method==2) // just by sorting
		{
			thrust::sort(first,last);
#ifdef DEBUG__CPS	
	time=stop();
	//printf("Elapsed time: selection by GPU sorting:   %lf\n",time);		
#endif				
			return *(first + k-1);
		}

		Tbase min_x, max_x;
		Tsum sum;
		
		ComputeMaxMinSum(first, last, max_x, min_x,sum);
		
		if(k==1) return min_x;
		if(k==n) return max_x;
		if(max_x==min_x) return min_x;
		
// if we are at risk of loss of precision, make a monotone transformation of the data		
		Tbase LARGE_NUMBER;
		if(sizeof(LARGE_NUMBER)<=4 )LARGE_NUMBER=LARGE_NUMBER_FLOAT;
		else LARGE_NUMBER=LARGE_NUMBER_DOUBLE;
		
		if((max_x-min_x) > LARGE_NUMBER) {
			transform(first,last,min_x);
			transformedcoefficient=min_x; transformed=1;
			max_x=log(MULT*(max_x-min_x)+1.0); min_x=0; sum=-10e10; // prompt to recalculate the sum
#ifdef DEBUG__CPS
		printf("transformation: now the range is   %lf %lf and coefficient is \n",min_x, max_x, transformedcoefficient);		
#endif
		}

		Tbase bracketL=min_x;  
		Tbase bracketR=max_x;

		Tbase A;
		int left, right;	
			
// do cutting plane iterations		
		int ret=mincp( min_x, max_x, sum,  CPiters,  &A, 
				&bracketL, &bracketR, &left, &right,first,last, n, k);  

#ifdef DEBUG__CPS
		printf("after CP: exit code %d, pivot interval  %lf %lf \n",ret,bracketL, bracketR);		
#endif
		
		if(ret==-1 ) { // chose an element above A
			A=ComputeMinLargerA(first,last,A);
		}
		else if(ret==1 || ret==0) { // choose an alement just below A
			A=ComputeMaxSmallerA(first,last,A);
		} else if(ret==3) // sort the pivot interval
			ComputeBySorting(first,last,A, bracketL, bracketR, 
							left, right, k-1 );
			
		if(transformed)
			A=transformback(A,MULT, transformedcoefficient);
			
#ifdef DEBUG__CPS	
	time=stop();
	printf("Elapsed time by cp_select:   %lf\n",time);		
#endif				
			
		return A;	
	}				
		
	
  // Call this for safety, CUDA may sometimes fail to exit gracefully by itself.
	void cleanUp()
	{
		 cudaThreadExit();
	}			
/*========================================================================*/


/* This program is to be called from .cu files. It copies data to GPU and calls median_min.
Cannot be called from cpp files (link errors).
*/
	template<typename InputIterator>
	typename thrust::iterator_traits<InputIterator>::value_type
	median_min_host(InputIterator first, InputIterator last,   int CPiters,  int method, int k=0)
	{
			typedef typename thrust::iterator_traits<InputIterator>::value_type Tbase;
			thrust::device_vector<Tbase> DataD(last-first);
			thrust::copy(first,last,DataD.begin());
			//DataD=first;
			Tbase res=median_min(DataD.begin(), DataD.end(),    CPiters,   method, k);
			DataD.resize(0);
			return res;
			
	}
	
/* These two functions copy vector D from CPU to GPU and call median_min. They can be called from .cpp files
*/	
	float median_min_float( thrust::host_vector<float> &D,      int n, int CPiters,  int method,  int k=0)
	{
		typedef float T;
		thrust::device_vector<T> DataD = D;
		T res=median_min(DataD.begin(), DataD.end(),    CPiters,   method,k);
		return res;
	
	}
	double  median_min_double( thrust::host_vector<double> &D,  int n, int CPiters,  int method, int k=0)
	{
		typedef double T;
		thrust::device_vector<T> DataD = D;
		T res=median_min(DataD.begin(), DataD.end(),    CPiters,   method, k);
		return res;
	}	

	

} // namespace


