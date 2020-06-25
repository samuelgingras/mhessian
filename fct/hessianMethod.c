#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "state.h"
#include "errors.h"
#include "model.h"
#include "RNG.h"

#define TRUE    1
#define FALSE   0

// hessianMethod( seed )
// hmout = hessianMethod( model, data, theta, ... )

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

        int seed = mxGetScalar(prhs[0]);
        rng_init_rand( (unsigned long)seed );
    }
    else 
    {
        // Check input and output arguments
        ErrMsgTxt( nrhs >= 3,
        "Invalid inputs: Three input arguments expected");
        
        ErrMsgTxt( nlhs == 1,
        "Invalid outputs: One or Two output argument expected");
                
        // Assign model
        Observation_model *model = assignModel( prhs[0] );

        // Allocate memory structure
        Theta *theta = thetaAlloc();
        State *state = stateAlloc();
        Data *data = dataAlloc();

        // Initialize
        model->initializeModel();
        model->initializeData( prhs[1], data );
        model->initializeTheta( prhs[2], theta );
        
        // Initialize mxState
        mxArray *mxState = mxStateAlloc(theta->alpha->n, model, state);
        
        // TODO: change function to check input compatibility
        // Set model specific function ?

        // Check observation data
        ErrMsgTxt( data->m == theta->alpha->n,
        "Invalid input argument: incompatible vector length");
        
        
        // Create/set field pointer for output structure
        mxArray *x = mxGetField(mxState,0,"x");
        mxArray *x_alC = mxGetField(mxState,0,"x_mode");
        mxArray *lnp_y = mxCreateDoubleMatrix(1,1,mxREAL);
        mxArray *lnp_x = mxCreateDoubleMatrix(1,1,mxREAL);
        mxArray *lnq_x = mxCreateDoubleMatrix(1,1,mxREAL);

        // Check computation options
        int iter;
        int isDraw = TRUE;
        if( (nrhs % 2) != 1  )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidRHS",
                "Set options: Pair of input by option required.");

        // Check GuessMode option
        for( iter=3; iter < nrhs; iter+=2 )
        {
            if( !mxIsChar(prhs[iter]) )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );

            if( strcmp(opt, "GuessMode") == 0)
            {
                if( !mxIsDouble(prhs[iter+1]) )
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GuessMode option: Double vector required.");

                if( mxGetM(prhs[iter+1]) != state->n )
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GuessMode option: Incompatible vector length.");

                state->guess_alC = TRUE;
                memcpy(state->alC, mxGetDoubles(prhs[iter+1]), state->n * sizeof(double));
            }
            mxFree(opt);
        }

        // Execute HESSIAN method
        // (1) Find conditional mode
        compute_alC_all(model, theta, state, data);

        // TODO: Add DoDiagnostic option here
        // ...

        // Check EvalAtState or EvalAtMode option
        for( iter=3; iter < nrhs; iter+=2 )
        {
            if( !mxIsChar(prhs[iter]) )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL )
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );

            if( strcmp(opt, "EvalAtMode") == 0) 
            {
                if( !mxIsLogicalScalar(prhs[iter+1]) )
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtMode option: Logical scalar required.");

                if( mxIsLogicalScalarTrue(prhs[iter+1]) )
                {  
                    isDraw = FALSE;
                    memcpy(state->alpha, state->alC, state->n * sizeof(double));
                }
            }
            else if( strcmp(opt, "EvalAtState") == 0)
            {
                if( !mxIsDouble(prhs[iter+1]) )
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtState: Double vector required.");

                if( mxGetM(prhs[iter+1]) != state->n )
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtState: Incompatible column vector.");

                isDraw = FALSE;
                memcpy(state->alpha, mxGetDoubles(prhs[iter+1]), state->n * sizeof(double));
            }
            mxFree(opt);
        }

        // (2) Draw state and evaluate proposal log-likelihood
        draw_HESSIAN( isDraw, model, theta, state, data, mxGetPr(lnq_x) );
        
        // (3) Evaluate state prior log-likelihood
        alpha_prior_eval( theta->alpha, state->alpha, mxGetPr(lnp_x) );
        
        // (4) Evaluate observation conditional log-likelihood given the states
        model->log_f_y__theta_alpha( state->alpha, theta->y, data, mxGetPr(lnp_y) );
        
        // TODO: Add GradHess option here and update output structure
        // ...

        // Create matlab output structure
        const char *field_names[] = {"x","x_mode","lnp_y__x","lnp_x","lnq_x__y"};
        
        plhs[0] = mxCreateStructMatrix(1,1,5,field_names);
        mxSetField(plhs[0],0,"x", mxDuplicateArray(x));
        mxSetField(plhs[0],0,"x_mode", mxDuplicateArray(x_alC));
        mxSetField(plhs[0],0,"lnp_y__x", lnp_y);
        mxSetField(plhs[0],0,"lnp_x", lnp_x);
        mxSetField(plhs[0],0,"lnq_x__y", lnq_x);
        
        // Return mxState if second output argument
        if( nlhs == 2 )
            plhs[1] = mxState;  
    }
}


