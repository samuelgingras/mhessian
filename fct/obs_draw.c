#include <stdlib.h>
#include "mex.h"
#include "state.h"
#include "model.h"
#include "errors.h"


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments	
    ErrMsgTxt( nrhs == 2,
    "Invalid inputs: Two input arguments expected");
    
    ErrMsgTxt( nlhs < 3,
    "Invalid outputs: One or Two output argument expected");
    
    // Assign model
    Observation_model *model = assign_model(prhs[1]);
    model->initialize();
    
    // Initialize parameter
    Parameter *theta_y = (Parameter *) mxMalloc(sizeof(Parameter));
    model->initializeParameter(prhs[1], theta_y);
    
    // Check state vector and Initialize data
    Data *data = (Data *) mxMalloc(sizeof(Data));
    mxCheckVector(prhs[0]);
    data->n = mxGetM(prhs[0]);
    
    // Prepare output argument and set pointer
    plhs[0] = mxCreateDoubleMatrix((mwSize)data->n,1,mxREAL);
    data->y = mxGetPr(plhs[0]);
    
    // Draw observations
    model->draw_y__theta_alpha(mxGetPr(prhs[0]), theta_y, data);
    
    // // Evaluate likelihood if output argument
    // if( nlhs == 2 ) {
    //     plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    //     model->log_f_y__theta_alpha(mxGetPr(prhs[0]), theta_y, data, mxGetPr(plhs[1]));
    // }
}