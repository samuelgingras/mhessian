#include <stdlib.h>
#include "mex.h"
#include "alpha_univariate.h"
#include "state.h"
#include "errors.h"
#include "model.h"


#define TRUE    1
#define FALSE   0

// Field of output structure (same for state_post functions)
// Put somewhere else?
const char *field_names[] = {
    "x",
    "x_mode",
    "lnp_y__x",
    "lnp_x",
    "lnq_x__y"
};


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments
    ErrMsgTxt( nrhs == 3 || nrhs == 4,
    "Invalid inputs: Three or Four input arguments expected");
    
    ErrMsgTxt( nlhs < 3,
    "Invalid outputs: One or Two output argument expected");
    
    // Check strucutre input
    for(int i=0; i<nrhs; i++)
        mxCheckStruct(prhs[i]);
    
    // Assign model
    Observation_model *model = assign_model(prhs[1]);
    model->initialize();
    
    // Initialize theta
    Theta *theta = mxThetaAlloc();
    model->initializeParameter(prhs[1], theta->y);
    initializeStateParameter(prhs[0], theta->alpha);
    
    // Initialize State
    State *state = (State *) mxMalloc( sizeof(State) ); 
    mxArray *mxState = mxStateAlloc(theta->alpha->n, model, state);
    setDefaultOptions(state);
    
    // Read Data
    Data *data = (Data *) mxMalloc( sizeof(Data) );
    model->read_data(prhs[2], data);
    
    // Check observation data
    ErrMsgTxt( data->m == theta->alpha->n,
    "Invalid input argument: incompatible vector length");
    
    // Read options
    if( nrhs == 4 )
        readComputationOptions(prhs[3], state);
    
    // Create/set field pointer for output structure
    mxArray *x = mxGetField(mxState,0,"x");
    mxArray *x_alC = mxGetField(mxState,0,"x_mode");
    mxArray *lnp_y = mxCreateDoubleMatrix(1,1,mxREAL);
    mxArray *lnp_x = mxCreateDoubleMatrix(1,1,mxREAL);
    mxArray *lnq_x = mxCreateDoubleMatrix(1,1,mxREAL);
    
    // Execute HESSIAN method
    // (1) Find conditional mode
    compute_alC_all(model, theta, state, data);
    
    // (2) Draw state and evaluate proposal log-likelihood
    draw_HESSIAN(TRUE, model, theta, state, data, mxGetPr(lnq_x));
    
    // (3) Evaluate state prior log-likelihood
    alpha_prior_eval(theta->alpha, state->alpha, mxGetPr(lnp_x));
    
    // (4) Evaluate observation conditional log-likelihood given the states
    model->log_f_y__theta_alpha(state->alpha, theta->y, data, mxGetPr(lnp_y));
    
    // Create matlab output structure
    plhs[0] = mxCreateStructMatrix(1,1,5,field_names);
    mxSetField(plhs[0],0,field_names[0], mxDuplicateArray(x));
    mxSetField(plhs[0],0,field_names[1], mxDuplicateArray(x_alC));
    mxSetField(plhs[0],0,field_names[2], lnp_y);
    mxSetField(plhs[0],0,field_names[3], lnp_x);
    mxSetField(plhs[0],0,field_names[4], lnq_x);
    
    // Return mxState if second output argument
    if( nlhs == 2 )
        plhs[1] = mxState;
}