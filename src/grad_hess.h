#include "mex.h"

#ifndef MEX_GRAD_HESS
#define MEX_GRAD_HESS

#define p_len 4

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

void compute_grad_Hess(
    int long_th, State *state, Theta *theta,  // Inputs
    double *grad, double *Hess, double *Var,  // Outputs
    double *xp, double *L_mu_mu_mu
);
#endif