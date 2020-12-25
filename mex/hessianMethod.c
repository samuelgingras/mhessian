#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "grad_hess.h"
#include "state.h"
#include "errors.h"
#include "model.h"
#include "RNG.h"

#define TRUE    1
#define FALSE   0


// Call for seed of rng:
// hessianMethod( seed )

// Call for computation
// hmout = hessianMethod( model, data, theta, ... )

// Call for computation with diagnostics variables
// [ hmout, state ] = hessianMethod( model, data, theta, ... )

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{

    // Set Seed
    if( nrhs == 1 ) {
        if( !mxIsScalar(prhs[0]) ) {
            mexErrMsgIdAndTxt( "mhessian:invalidInputs",
                "SetSeed: Scaler input required.");
        }

        // TODO: check if input is an integer
        // ...

        if( mxGetScalar(prhs[0]) < 0 ) {
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "SetSeed: Positive integer required.");
        }

        int seed = mxGetScalar(prhs[0]);
        rng_init_rand( (unsigned long)seed );
    }
    else {
        // Check input and output arguments
        ErrMsgTxt( nrhs >= 3,
        "Invalid inputs: Three input arguments expected");
        
        ErrMsgTxt( nlhs <= 2,
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
        mxArray *xC = mxGetField(mxState,0,"xC");
        mxArray *lnp_y = mxCreateDoubleMatrix(1,1,mxREAL);
        mxArray *lnp_x = mxCreateDoubleMatrix(1,1,mxREAL);
        mxArray *lnq_x = mxCreateDoubleMatrix(1,1,mxREAL);


        // (0) Parse options (computation and output options)
        int iter;                   // To parse pair of (opt, value)
        int isDraw = TRUE;          // Draw and Eval or Eval only
        int doGradHess = FALSE;     // Compute grad Hess approximation
        int long_th = TRUE;         // Size of theta for grad Hess


        // Check pair input for options
        if( (nrhs % 2) != 1  ) {
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:rhs", 
                "Options: Pair of input by option required.");
        }

        // Compute diagnostics if two output arguments
        // TODO: Update this option ...
        if( nlhs == 2 ) {
            state->compute_diagnostics = TRUE;  // Activate compute_diagnostics function
            plhs[1] = mxState;                  // Return all computation by-product
        }

        // Check DataAugmentation option
        for( iter=3; iter < nrhs; iter+=2 ) {
            if( !mxIsChar(prhs[iter]) ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");
            }

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );
            }

            if( strcmp(opt, "DataAugmentation") == 0) {
                if( !mxIsLogicalScalar(prhs[iter+1]) ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtMode option: Logical scalar required.");
                }
                theta->y->is_data_augmentation = mxIsLogicalScalarTrue(prhs[iter+1]);
            }
            mxFree(opt);
        }

        // Check GuessMode option
        for( iter=3; iter < nrhs; iter+=2 ) {
            if( !mxIsChar(prhs[iter]) ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");
            }

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );
            }

            if( strcmp(opt, "GuessMode") == 0) {
                if( !mxIsDouble(prhs[iter+1]) ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GuessMode option: Double vector required.");
                }
                if( mxGetM(prhs[iter+1]) != state->n ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GuessMode option: Incompatible vector length.");
                }
                state->guess_alC = TRUE;
                memcpy(state->alC, mxGetDoubles(prhs[iter+1]), state->n * sizeof(double));
            }
            mxFree(opt);
        }


        // (1) Find conditional mode and compute derivatives
        compute_alC_all(model, theta, state, data);


        // Check EvalAtState or EvalAtMode option
        for( iter=3; iter < nrhs; iter+=2 ) {
            if( !mxIsChar(prhs[iter]) ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");
            }

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );
            }

            if( strcmp(opt, "EvalAtMode") == 0) {
                if( !mxIsLogicalScalar(prhs[iter+1]) ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtMode option: Logical scalar required.");
                }
                if( mxIsLogicalScalarTrue(prhs[iter+1]) ) {  
                    isDraw = FALSE;
                    memcpy(state->alpha, state->alC, state->n * sizeof(double));
                }
            }
            else if( strcmp(opt, "EvalAtState") == 0) {
                if( !mxIsDouble(prhs[iter+1]) ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtState: Double vector required.");
                }
                if( mxGetM(prhs[iter+1]) != state->n ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "EvalAtState: Incompatible column vector.");
                }
                isDraw = FALSE;
                memcpy(state->alpha, mxGetDoubles(prhs[iter+1]), state->n * sizeof(double));
            }
            mxFree(opt);
        }

        // Check gradHess option
        for( iter=3; iter < nrhs; iter+=2 ) {
            if( !mxIsChar(prhs[iter]) ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                    "Set options: String input required.");
            }

            char *opt = mxArrayToString( prhs[iter] );
            if( opt == NULL ) {
                mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                    "Error reading computation option." );
            }

            if( strcmp(opt, "GradHess") == 0 ) {
                if( !theta->alpha->is_grad_hess ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GradHess: Unavailable theta specification."); 
                }
                if( !mxIsChar(prhs[iter+1]) ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GradHess option: 'Long' or 'Short' option required.");
                }

                char *value = mxArrayToString( prhs[iter+1] );
                if( value == NULL ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
                        "Error reading computation option." );
                }
                if( strcmp( value, "Short") == 0 ) {
                    long_th = FALSE;
                }
                else if( strcmp( value, "Long" ) != 0 ) {
                    mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                        "GradHess option: 'Long' or 'Short' option available.");
                }
                doGradHess = TRUE;
                mxFree(value);
            }
            mxFree(opt);
        }

        // (3) Draw state and/or evaluate proposal log-likelihood
        draw_HESSIAN( isDraw, model, theta, state, data, mxGetPr(lnq_x) );
        
        // (4) Evaluate states prior log-likelihood
        alpha_prior_eval( theta->alpha, state->alpha, mxGetPr(lnp_x) );
        
        // (5) Evaluate observations conditional log-likelihood given the states
        model->log_f_y__theta_alpha( state->alpha, theta->y, data, mxGetPr(lnp_y) );
        

        // Create MATLAB output structure
        if( doGradHess ) {

            mwSize dim_th = long_th ? 3 : 2;
            mwSize dims[3];
            dims[0] = dims[1] = dims[2] = dim_th;
            mxArray *grad = mxCreateDoubleMatrix(dim_th, 1, mxREAL);
            mxArray *Hess = mxCreateDoubleMatrix(dim_th, dim_th, mxREAL);
            mxArray *Var = mxCreateDoubleMatrix(dim_th, dim_th, mxREAL);
            mxArray *d1n_sum = mxCreateDoubleMatrix(1, 1, mxREAL);
            mxArray *dt_sum = mxCreateDoubleMatrix(1, 1, mxREAL);
            mxArray *d11nn_sum = mxCreateDoubleMatrix(1, 1, mxREAL);
            mxArray *dtt_sum = mxCreateDoubleMatrix(1, 1, mxREAL);
            mxArray *dttp_sum = mxCreateDoubleMatrix(1, 1, mxREAL);
            mxArray *g_cum = mxCreateDoubleMatrix(state->n, 5, mxREAL);
            // Restore next line if implementinig 3rd derivative information
            // mxArray *T = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);

            // Compute gradHess approximation
            compute_grad_Hess( long_th, state, theta,
                mxGetPr(grad), mxGetPr(Hess), mxGetPr(Var),
                mxGetPr(d1n_sum), mxGetPr(dt_sum), mxGetPr(d11nn_sum), mxGetPr(dtt_sum), mxGetPr(dttp_sum),
                mxGetPr(g_cum) );

            // Set field names
            const char *field_hmout[] = {"x", "xC", "lnp_y__x", "lnp_x", "lnq_x__y", "q_theta"};
            const char *field_gradHess[] = {"grad", "Hess", "Var",
                                            "d1n_sum", "dt_sum", "d11nn_sum", "dtt_sum", "dttp_sum", "g_cum"};
            
            // Create gradHess output structure
            mxArray *q_theta = mxCreateStructMatrix(1, 1, 9, field_gradHess);
            mxSetField(q_theta, 0, "grad", grad);
            mxSetField(q_theta, 0, "Hess", Hess);
            mxSetField(q_theta, 0, "Var", Var);
            mxSetField(q_theta, 0, "d1n_sum", d1n_sum);
            mxSetField(q_theta, 0, "dt_sum", dt_sum);
            mxSetField(q_theta, 0, "d11nn_sum", d11nn_sum);
            mxSetField(q_theta, 0, "dtt_sum", dtt_sum);
            mxSetField(q_theta, 0, "dttp_sum", dttp_sum);
            mxSetField(q_theta, 0, "g_cum", g_cum);

            // Restore next line if implementing 3rd derivative information
            // mxSetField(q_theta, 0, "T", T);


            // Create MATLAB output structure
            plhs[0] = mxCreateStructMatrix(1, 1, 6, field_hmout);
            mxSetField(plhs[0], 0, "x", mxDuplicateArray(x));
            mxSetField(plhs[0], 0, "xC", mxDuplicateArray(xC));
            mxSetField(plhs[0], 0, "lnp_y__x", lnp_y);
            mxSetField(plhs[0], 0, "lnp_x", lnp_x);
            mxSetField(plhs[0], 0, "lnq_x__y", lnq_x);
            mxSetField(plhs[0], 0, "q_theta", q_theta);

        }
        else {

            // Set field names
            const char *field_hmout[] = {"x", "xC", "lnp_y__x", "lnp_x", "lnq_x__y"};
            
            // Create MATLAB output structure
            plhs[0] = mxCreateStructMatrix(1, 1, 5, field_hmout);
            mxSetField(plhs[0], 0, "x", mxDuplicateArray(x));
            mxSetField(plhs[0], 0, "xC", mxDuplicateArray(xC));
            mxSetField(plhs[0], 0, "lnp_y__x", lnp_y);
            mxSetField(plhs[0], 0, "lnp_x", lnp_x);
            mxSetField(plhs[0], 0, "lnq_x__y", lnq_x);

        }
    }
}


