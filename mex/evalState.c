#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "state.h"
#include "errors.h"

// lnp_x = evalState( x, theta )

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments
    ErrMsgTxt( nrhs == 2,
    "Invalid inputs: Two input arguments expected");
    
    ErrMsgTxt( nlhs <= 1,
    "Invalid outputs: One output argument expected");
    
    // Allocate memory and initialize state parameters
    Theta *theta = thetaAlloc();
    initializeThetaAlpha( mxGetField( prhs[1], 0, 'x' ), theta->alpha );
    
    // Check state vector
    if( !mxIsDouble(prhs[0]) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Vector of double required.");

    if( mxGetN(prhs[0]) != 1 )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Column vector required.");

    if( mxGetM(prhs[0]) != theta->alpha->n )
    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
        "Incompatible vector length.");
    
    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix( 1, 1, mxREAL );
    
    // Evaluate state
    alpha_prior_eval( theta->alpha, mxGetDoubles(prhs[0]), mxGetDoubles(plhs[0]) );
}