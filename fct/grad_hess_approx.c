#include <string.h>
#include "mex.h"
#include "grad_hess.h"

void ErrMsgTxt(bool assertion, const char *text)
{
    if(!assertion)
        mexErrMsgTxt(text);
}

void mexFunction(int nlhs, mxArray *plhs[], const int nrhs, const mxArray *prhs[])
{
    // Check number of input/output argument
    ErrMsgTxt( nrhs == 3,
    "Invalid inputs: Three input arguments expected");
    
    ErrMsgTxt( nlhs == 3,
    "Invalid outputs: Three output arguments expected");
    
    // Check if structure
    ErrMsgTxt( mxIsStruct(prhs[0]),
    "Invalid inputs: structure argument expected");
    
    ErrMsgTxt( mxIsStruct(prhs[1]),
    "Invalid inputs: structure argument expected");
    
    // Set pointer to each field (theta input)
    mxArray *pr_N = mxGetField(prhs[0],0,"N");
    mxArray *pr_d = mxGetField(prhs[0],0,"d");
    mxArray *pr_mu = mxGetField(prhs[0],0,"mu");
    mxArray *pr_phi = mxGetField(prhs[0],0,"phi");
    mxArray *pr_omega = mxGetField(prhs[0],0,"omega");
    
    // Check for missing field (theta input)
    ErrMsgTxt( pr_N != NULL,
    "Invalid input argument: number of observation expected");
    ErrMsgTxt( pr_d != NULL || pr_mu != NULL,
    "Invalid input argument: mean/intercept parameter expected");
    ErrMsgTxt( pr_phi != NULL,
    "Invalid input argument: persitence parameter expected");
    ErrMsgTxt( pr_omega != NULL,
    "Inbalid input argument: precision parameter expected");
    
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
    
    // Prepare output arguments
    plhs[0] = mxCreateDoubleMatrix(3,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(3,3,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(3,3,mxREAL);
    
    // Set pointers
    double *u = mxGetPr(prhs[2]);
    double *grad = mxGetPr(plhs[0]);
    double *Hess = mxGetPr(plhs[1]);
    double *var = mxGetPr(plhs[2]);
    
    
    compute_grad_Hess(prhs[1], n, mu, phi, omega, u, grad, Hess, var);
}