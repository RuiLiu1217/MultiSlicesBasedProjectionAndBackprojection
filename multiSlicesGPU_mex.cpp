/*
 * Wake Forest Health Sciences & University of Massachusetts Lowell
 * Organization: 
 *  Wake Forest Health Sciences
 *
 * multiSlicesGPU_mex.cpp
 * Matlab mex gateway routine for the GPU based multi-slice distance-driven
 * fan-beam projector
 *
 * author: Rui Liu (Wake Forest Health Sciences)
 * date: 2016.09.25
 * version: 1.0
 */

#include "mex.h"
#include "matrix.h"
#include <cstring>
#include <iostream>

#include "multiSlices_ker.h"
typedef unsigned char byte;

extern "C"
void DD2_multiGPU(
		float* hvol, // the pointer to the image
		float* hprj, // the pointer to the projection (SLN, DNU, PN) order
		const int method, // Control to use forward projection or backprojection
		const float x0, const float y0, //position of the initial source
		float* xds, float* yds, // distribution of the detector cells
		const int DNU, // Number of detector cells
		const int SLN, // Number of slices to be projected or backprojected
		const float imgXCenter, const float imgYCenter, //Center of the image
		const int XN, const int YN, // pixel number of the image
		const float dx, // size of the pixel
		float* hangs, // view angles (the size should be SLN * PN)
		int PN, // # of view angles
		byte* mask,
		int* startIdx,
		const int gpuNum);


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    
    int method = *((int*)mxGetData(prhs[0]));
    float x0 = *((float*)mxGetData(prhs[1]));
    float y0 = *((float*)mxGetData(prhs[2]));
    float* xds = (float*)mxGetPr(prhs[3]);
    float* yds = (float*)mxGetPr(prhs[4]);
    int DNU = *((int*)mxGetData(prhs[5]));
    int SLN = *((int*)mxGetData(prhs[6]));
    float imgXCenter = *((float*)mxGetData(prhs[7]));
    float imgYCenter = *((float*)mxGetData(prhs[8]));
    int XN = *((int*)mxGetData(prhs[9]));
    int YN = *((int*)mxGetData(prhs[10]));
    float dx = *((float*)mxGetData(prhs[11]));
    float* hangs = (float*)mxGetPr(prhs[12]);
    int PN = *((int*)mxGetData(prhs[13]));
    byte* mask = (byte*)mxGetPr(prhs[14]);
    int* startIdx = (int*)mxGetPr(prhs[15]);
    int gpuNum = *((int*)mxGetData(prhs[16]));
    
    if(method == 0 || method == 2) // projection
    {
        const mwSize dims[] = {SLN, DNU, PN};
        plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);    
        float* hprj = (float*)mxGetPr(plhs[0]);
        float* hvol = (float*)mxGetPr(prhs[17]);
        DD2_multiGPU(hvol, hprj, method, x0, y0, xds, yds, 
            DNU, SLN, imgXCenter, imgYCenter, XN, YN, dx, 
                hangs, PN, mask, startIdx, gpuNum);
    }
    else if(method == 1 || method == 3)
    {
        const mwSize dims[] = {SLN, XN, YN};
        plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);    
        float* hvol = (float*)mxGetPr(plhs[0]);
        float* hprj = (float*)mxGetPr(prhs[17]);
        DD2_multiGPU(hvol, hprj, method, x0, y0, xds, yds, 
            DNU, SLN, imgXCenter, imgYCenter, XN, YN, dx, 
                hangs, PN, mask, startIdx, gpuNum);
    }
    else
    {
        std::cerr<<"Unknown methods routine\n";
        exit(-1);
    }
    
   
    
}
