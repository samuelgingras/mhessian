#include <string.h>
#include "mex.h"
#include "new_grad_hess.h"


void ErrMsgTxt(bool assertion, const char *text)
{
    if(!assertion)
        mexErrMsgTxt(text);
}

void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray *prhs[])
{
    
    ErrMsgTxt( nrhs == 3,
    "Invalid inputs: Three input arguments expected");
    
    ErrMsgTxt( nlhs == 3,
    "Invalid outputs: Three output arguments expected");
    
    // Check if structure
    ErrMsgTxt( mxIsStruct(prhs[0]),
    "Invalid inputs: structure argument expected");
    
    ErrMsgTxt( mxIsStruct(prhs[1]),
    "Invalid inputs: structure argument expected");
    
    // Check if nested structure

    mxArray *pr_N, *pr_d, *pr_mu, *pr_phi, *pr_omega;
    mxArray *pr_theta_x = mxGetField( prhs[0], 0, "x" );
    if( pr_theta_x == NULL )
    {
        pr_N = mxGetField( prhs[0], 0, "N" );
        pr_d = mxGetField( prhs[0], 0, "d" );
        pr_mu = mxGetField( prhs[0], 0, "mu" );
        pr_phi = mxGetField( prhs[0], 0, "phi" );
        pr_omega = mxGetField( prhs[0], 0, "omega" );

    }
    else
    {
        pr_N = mxGetField( pr_theta_x, 0, "N" );
        pr_d = mxGetField( pr_theta_x, 0, "d" );
        pr_mu = mxGetField( pr_theta_x, 0, "mu" );
        pr_phi = mxGetField( pr_theta_x, 0, "phi" );
        pr_omega = mxGetField( pr_theta_x, 0, "omega" );
    }

    // Check for missing field (theta input)
    ErrMsgTxt( pr_N != NULL,
    "Invalid input argument: number of observation expected");
    ErrMsgTxt( pr_d != NULL || pr_mu != NULL,
    "Invalid input argument: mean/intercept parameter expected");
    ErrMsgTxt( pr_phi != NULL,
    "Invalid input argument: persistence parameter expected");
    ErrMsgTxt( pr_omega != NULL,
    "Invalid input argument: precision parameter expected");
    
    // Read number of observation
    ErrMsgTxt( mxIsScalar(pr_N),
    "Invalid input argument: scalar value expected");
    int n = mxGetScalar(pr_N);
    
    // Read phi parameter
    ErrMsgTxt( mxIsScalar(pr_phi),
    "Invalid input argument: scalar value expected");
    double phi = mxGetScalar(pr_phi);
    
    // Read omega parameter
    ErrMsgTxt( mxIsScalar(pr_omega),
    "Invalid input argument: scalar value expected");
    double omega = mxGetScalar(pr_omega);
    
    // Read mean/intercept parameter
    double *mu = (double *) mxCalloc(n, sizeof(double));
    if( pr_mu != NULL ){
        if( mxIsScalar(pr_mu) )
            for(int i=0;i<n;i++) {mu[i] = mxGetScalar(pr_mu);}
        else
            memcpy(mu, mxGetPr(pr_mu), n * sizeof(double));
    }
    if( pr_d != NULL ){
        double *d = mxGetPr(pr_d);
        mu[0] = d[0] + mu[0];
        for(int i=1;i<n;i++)
            mu[i] = d[i] + mu[i] - phi * d[i-1];
    }
    
    double *p_long_th = mxGetPr(prhs[2]);
    int long_th = p_long_th[0] != 0;
    int dim_th = long_th ? 3 : 2;

    // Prepare output arguments
    plhs[0] = mxCreateDoubleMatrix(dim_th, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(dim_th, dim_th, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(dim_th, dim_th, mxREAL);
    
    // Set pointers
    double *grad = mxGetPr(plhs[0]);
    double *Hess = mxGetPr(plhs[1]);
    double *var = mxGetPr(plhs[2]);
    
    compute_new_grad_Hess(prhs[1], long_th, n, mu, phi, omega, grad, Hess, var);
}