#include "mex.h"

#ifndef MEX_GRAD_HESS
#define MEX_GRAD_HESS

void compute_grad_Hess(
        const mxArray *mxState,
        int n,
        double *mu,
        double phi,
        double omega,
        double *u,
        double *grad,
        double *Hess,
        double *var
    );
#endif