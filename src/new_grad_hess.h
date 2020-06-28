#include "mex.h"

#ifndef MEX_GRAD_HESS
#define MEX_GRAD_HESS

typedef struct {
    double coeff_11;
    double coeff_tt;
    double coeff_ttp;
} Q_mat;

typedef struct {
    double coeff_1;
    double cooff_t;
} q_vec;

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