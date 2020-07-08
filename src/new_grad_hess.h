#include "mex.h"

#ifndef MEX_NEW_GRAD_HESS
#define MEX_NEW_GRAD_HESS

typedef struct {
    double Q_11;
    double Q_tt;
    double Q_ttp;
    double q_1;
    double q_t;
    double m_tm1[3];
    double m_t[3];
    double dQd;
    double qd;
} Q_term;

typedef struct {
    int i, j;
    double c_tm1[3];
    double c_t[3];
} C_term;

void compute_new_grad_Hess(
        const mxArray *mxState,
        int n,
        double *mu,
        double phi,
        double omega,
        double *grad,
        double *Hess,
        double *var
    );
#endif