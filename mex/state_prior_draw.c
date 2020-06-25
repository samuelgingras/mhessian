#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "state.h"
#include "errors.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments
    ErrMsgTxt( nrhs == 1,
    "Invalid inputs: One input argument expected");
    
    ErrMsgTxt( nlhs < 3,
    "Invalid outputs: One or Two output argument expected");
    
    // Initialize state parameter
    State_parameter *theta_alpha = (State_parameter *) mxMalloc(sizeof(State_parameter));
    initializeStateParameter(prhs[0], theta_alpha);
    
    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix((mwSize)theta_alpha->n,1,mxREAL);
    
    // Draw state and evaluate likelihood
    alpha_prior_draw(theta_alpha, mxGetPr(plhs[0]));
    
    // Evaluate likelihood if output argument
    if( nlhs == 2 ) {
        plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
        alpha_prior_eval(theta_alpha, mxGetPr(plhs[0]), mxGetPr(plhs[1]));
    }
}