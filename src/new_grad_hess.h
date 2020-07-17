#include "mex.h"

#ifndef MEX_NEW_GRAD_HESS
#define MEX_NEW_GRAD_HESS

#define p_len 5

typedef struct {
    double Q_11;
    double Q_tt;
    double Q_ttp;
    double q_1;
    double q_t;
    double m_tm1[p_len];
    double m_t[p_len];
    double dQd;
    double qd;
} Q_term;

typedef struct {
    int i, j;
    double c_tm1[p_len];
    double c_t[p_len];
} C_term;

void compute_new_grad_Hess(
        const mxArray *mxState,
        int long_th,
        int n,
        double *mu,
        double phi,
        double omega,
        double *grad,
        double *Hess,
        double *var
    );
#endif