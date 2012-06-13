10 April 2011
author Gleb Beliakov (gleb@deakin.edu.au)
CP_Select library (Cutting Plane selection algorithm). Freeware.

This is the library cp_select Version 1.0 for parallel selection on GPU and 
a test program illustrating it use.
Comments on the usage of cp_select are in cp_select.h file.

Both sources depend on Thrust library, which should be installed first, and of course, on CUDA.

compile both files using ./compileme script
or any other appropriate method (eg. converting it to a project for Visual C++)

If you are interested in timing the selection methods, uncomment the line
//#define DEBUG__CPS 
in cp_select.cu file.

nvcc -O2 -c -arch=sm_20 cp_select.cu
is to compile the library only. The option -arch=sm_20 (or -arch=sm_13) is to  ensure 
doubles are not downgraded to floats



