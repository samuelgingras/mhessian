#include <stdlib.h>
#include "mex.h"
#include "state.h"
#include "model.h"
#include "errors.h"

// lnp_y__x = evalSample( y, x, model, theta )

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments 
    ErrMsgTxt( nrhs >= 3 ,
    "Invalid inputs: Two input arguments expected");
    
    ErrMsgTxt( nlhs <= 2,
    "Invalid outputs: One or Two output argument expected");
    
    // Allocate memory 
    Theta *theta = thetaAlloc();
    Data *data = dataAlloc();

    // Assign and initialize model
    Observation_model *model = assignModel( prhs[2] );
    model->initializeModel();
    model->initializeData( prhs[0], data );

    // Read model parameters
    if( model->n_theta > 0 )
    {
        if( nrhs != 4 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Four input argument required.");

        if( !mxIsStruct(prhs[3]) )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Model parameters: Structure input required.");

        mxArray *pr_theta_y = mxGetField( prhs[3], 0, "y" );

        if( pr_theta_y == NULL )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Structure input: Field 'y' required.");

        model->initializeParameter( pr_theta_y, theta->y );
    }

    // Check state vector
    if( !mxIsDouble(prhs[1]) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Vector of double required.");

    if( mxGetN(prhs[1]) != 1 )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Column vector required.");

    if( mxGetM(prhs[1]) != data->n )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Incompatible vector length.");

    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix( 1, 1, mxREAL ); 

    // Draw observations
    model->log_f_y__theta_alpha( mxGetDoubles(prhs[1]) , theta->y, data, mxGetDoubles(plhs[0]) );
}