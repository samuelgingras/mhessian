#include "mex.h"
#include "state.h"

#ifndef MEX_MODEL
#define MEX_MODEL

Observation_model *assign_model(const mxArray *prhs);

#endif