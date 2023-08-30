#include <stdlib.h>
#include <string.h>
#include "mex.h"

#ifndef MEX_ERRORS
#define MEX_ERRORS

extern char invalid_input[];
extern char missing_input[];

void ErrMsgTxt(bool assertion, const char *text);
void mxCheckStruct(const mxArray *prhs);
void mxCheckVector(const mxArray *prhs);
void mxCheckVectorSize(int n, const mxArray *prhs);

#endif