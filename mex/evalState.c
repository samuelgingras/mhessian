#include <stdlib.h>
#include "mex.h"
#include "x_univariate.h"
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
    
    // Check if structure input
    if( !mxIsStruct(prhs[1]) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input required.");

    // Allocate memory and initialize state parameters
    Theta *theta = thetaAlloc();
    if( mxGetField( prhs[1], 0, "x" ) != NULL )
        initializeThetax( mxGetField( prhs[1], 0, "x" ), theta->x );
    else
        initializeThetax( prhs[1], theta->x );

    // Check state vector
    if( !mxIsDouble(prhs[0]) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Vector of double required.");

    if( mxGetN(prhs[0]) != 1 )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Column vector required.");

    if( mxGetM(prhs[0]) != theta->x->n )
    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
        "Incompatible vector length.");
    
    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix( 1, 1, mxREAL );
    
    // Evaluate state
    x_prior_eval( theta->x, mxGetDoubles(prhs[0]), mxGetDoubles(plhs[0]) );
}