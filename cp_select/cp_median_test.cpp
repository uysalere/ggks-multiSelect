#include <algorithm>
#include <iostream>
#include <math.h>

#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;


#include "cp_select.h"
#include <thrust/host_vector.h>

/* This sample program illustrates the use of cp_select library. It generates random data of desired length
of types float and double, and then selects the median (for floats) and the desired order statistic (for doubles)
by using either GPU cp_select method (method=1) or by sorting on GPU (method=2).

compile this program using the script ./compileme (on linux) or an appropriate command on Windows, and run 
with the options
testmed length_of_data method [k]
(k means k-th order statisic)

e.g. 
./testmed 20000 1 25


In cp_select you can uncomment the line
//#define DEBUG__CPS

in this case the cp_select algorithm will also measure and report the wall time taken
to perform selection on GPU.

Please refer to the accompanying paper for details of the algorithm and underlying theory.

Depends on: Thrust and CUDA libraries

This is freeware.
*/

using namespace cp_select;


double urand() {
return (1.0*rand())/(RAND_MAX);
}

template <typename T>
void generatedata( unsigned int numElements, thrust::host_vector<T> &DataArray )
{
	srand(17);
	unsigned int i;
    for(i=0;i<numElements; i++) DataArray[i]=(T)urand();
}


	
int main(int argc, char *argv[]) 
{  
	int numElements  =  131072;// = 8192; //8192,131072,1048576,16777216,8388608,134217728,
	if ( argc < 3 ) 
	{
		cout <<"Usage ./testmedian numElements method = 1 or 2>  \nNow Exiting\n";
		exit(0);
	}
	numElements = atoi( argv[1] );
	int method = atoi( argv[2] );
	int k=10;
	if(argc==4) k = atoi( argv[3] );

	cout<<"selecting the median of "<<numElements<<" elements on GPU"<<endl;


	warmUp();
	double res;


	thrust::host_vector<float> Data(numElements);
	generatedata<float>(numElements, Data);

	res=median_min_float(Data, numElements, 7, method);
	cout<<"the median (float) is "<< res<< endl;

	thrust::host_vector<double> DataDouble(numElements);
	generatedata<double>(numElements, DataDouble);

	res=median_min_double(DataDouble, numElements, 7, method, k);
	cout<<"the k-th order statistic (double) is "<< res<< endl;

	cleanUp();


  return 0;	
}

