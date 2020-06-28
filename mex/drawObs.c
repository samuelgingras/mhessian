#include <stdlib.h>
#include "mex.h"
#include "state.h"
#include "model.h"
#include "errors.h"
#include "RNG.h"

// drawSample( seed )
// y = drawSample( x, model, theta )

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Set Seed
    if( nrhs == 1 )
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
        // Check input and output arguments 
        ErrMsgTxt( nrhs >= 2,
        "Invalid inputs: Two or three input arguments required.");
        
        ErrMsgTxt( nlhs <= 1,
        "Invalid outputs: One argument required.");
        
        // Allocate memory 
        Theta *theta = thetaAlloc();
        Data *data = dataAlloc();

        // Assign and initialize model
        Observation_model *model = assignModel( prhs[1] );
        model->initializeModel();

        // Read model parameters
        if( model->n_theta > 0 )
        {
            if( nrhs != 3 )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Three input argument required.");

            if( !mxIsStruct(prhs[2]) )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Model parameters: Structure input required.");

            mxArray *pr_theta_y = mxGetField( prhs[2], 0, "y" );

            if( pr_theta_y == NULL )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Structure input: Field 'y' required.");

            model->initializeParameter( pr_theta_y, theta->y );
        }


        // Check state vector
        if( !mxIsDouble(prhs[0]) )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Vector of double required.");

        if( mxGetN(prhs[0]) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector required.");

        // Prepare output argument
        plhs[0] = mxCreateDoubleMatrix( (mwSize)mxGetM(prhs[0]), 1, mxREAL ); 
        
        // Initialize data and set pointer to output argument
        data->n = mxGetM(prhs[0]);
        data->y = mxGetDoubles(plhs[0]);

        // Draw observations
        model->draw_y__theta_alpha( mxGetDoubles(prhs[0]), theta->y, data );
    }
}