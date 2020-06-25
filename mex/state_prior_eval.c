#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "state.h"
#include "errors.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments
    ErrMsgTxt( nrhs == 2,
    "Invalid inputs: Two input arguments expected");
    
    ErrMsgTxt( nlhs < 2,
    "Invalid outputs: One output argument expected");
    
    // Initialize state parameter
    State_parameter *theta_alpha = (State_parameter *) mxMalloc(sizeof(State_parameter));
    initializeStateParameter(prhs[1], theta_alpha);
    
    // Check state vector
    mxCheckVector(prhs[0]);
    mxCheckVectorSize(theta_alpha->n, prhs[0]);
    
    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    // Evaluate state
    alpha_prior_eval(theta_alpha, mxGetPr(prhs[0]), mxGetPr(plhs[0]));
}