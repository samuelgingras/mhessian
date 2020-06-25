#include <stdlib.h>
#include "mex.h"
#include "state.h"
#include "model.h"
#include "errors.h"


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // Check input and output arguments	
    ErrMsgTxt( nrhs == 3,
    "Invalid inputs: Three input arguments expected");
    
    ErrMsgTxt( nlhs < 2 ,
    "Invalid outputs: One output argument expected");
    
    // Assign model
    Observation_model *model = assign_model(prhs[1]);
    model->initialize();
    
    // Initialize parameter
    Parameter *theta_y = (Parameter *) mxMalloc( sizeof(Parameter) );
    model->initializeParameter(prhs[1], theta_y);
    
    // Initialize data
    Data *data = (Data *) mxMalloc( sizeof(Data) );
    model->read_data(prhs[2], data);
    
    // Check state vector
    mxCheckVector(prhs[0]);
    mxCheckVectorSize(data->n, prhs[0]);
    
    // Prepare output argument
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    // Evaluate likelihood
    model->log_f_y__theta_alpha(mxGetPr(prhs[0]),theta_y, data, mxGetPr(plhs[0]));
}