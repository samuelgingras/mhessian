#include <stdlib.h>
#include "mex.h"
#include "x_univariate.h"
#include "state.h"
#include "errors.h"
#include "RNG.h"

// drawState( seed )
// x = drawState( theta )

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{

    // Check input and output arguments
    ErrMsgTxt( nrhs == 1,
    "Invalid inputs: One input argument required.");
        
    ErrMsgTxt( nlhs <= 1,
    "Invalid outputs: One output argument required.");

    // Set Seed
    if( !mxIsStruct(prhs[0]) )
    {
        if( !mxIsScalar(prhs[0]) )
            mexErrMsgIdAndTxt( "mhessian:invalidInputs",
                "SetSeed: Scaler input required.");

        // TODO: check if input is an integer
        // ...

        if( mxGetScalar(prhs[0]) < 0 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "SetSeed: Positive integer required.");

        // Set Seed
        int seed = mxGetScalar(prhs[0]);
        rng_init_rand( (unsigned long)seed );
    }
    else
    {
        // Allocate memory and initialize state parameters
        Theta *theta = thetaAlloc();

        mxArray *pr_theta_x = mxGetField( prhs[0], 0, "x" );

        if( pr_theta_x == NULL )
            initializeThetax( prhs[0], theta->x );
        else
            initializeThetax( pr_theta_x, theta->x );
    
        // Prepare output argument
        plhs[0] = mxCreateDoubleMatrix( (mwSize)theta->x->n, 1, mxREAL );
    
        // Draw state vector
        x_prior_draw( theta->x, mxGetDoubles(plhs[0]) );    
    }
}