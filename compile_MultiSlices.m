%------------------------------------------------------------------------- 
% Wake Forest University Health Sciences
% Date: Sep. 25, 2016
% Routine: compile_MultiSlices.m
% Author
%	Rui Liu
% Organization: 
%  Wake Forest Health Sciences.
%-------------------------------------------------------------------------
% % 

system( '/usr/local/cuda/bin/nvcc -std=c++11 -Xcompiler -fopenmp -O3 --use_fast_math --compile -o multiSlices_ker.o  --compiler-options -fPIC  -I"/usr/local/MATLAB/R2015b/extern/extern/include " -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc "multiSlices_ker.cu" ' );

mex -v -largeArrayDims  COMPFLAGS="$COMPFLAGS -fopenmp -std=c++11" -L/usr/local/cuda/lib64 -lcudart -lgomp multiSlicesGPU_mex.cpp multiSlices_ker.o 
