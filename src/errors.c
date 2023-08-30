#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"

char invalid_input[] = "mhessian:invalidInput";
char missing_input[] = "mhessian:missingInput";

void ErrMsgTxt(bool assertion, const char *text)
{
    if(!assertion)
        mexErrMsgTxt(text);
}

void mxCheckStruct(const mxArray *prhs)
{
    ErrMsgTxt( mxIsStruct(prhs),
        "Invalid input argument: structure argument expected");
}

void mxCheckVector(const mxArray *prhs)
{
    ErrMsgTxt( mxIsDouble(prhs),
        "Invalid input argument: vector of double expected");
    ErrMsgTxt( mxGetN(prhs),
        "Invalid input argument: column vector expected");
}

void mxCheckVectorSize(int n, const mxArray *prhs)
{
    ErrMsgTxt( mxGetM(prhs) == n,
        "Invalid input argument: incompatible vector length");
}
