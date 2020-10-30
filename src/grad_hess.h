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
    double *d1n_sum,     // (x0_1-mu) + (x0_n-mu)
    double *dt_sum,      // (x0_2-mu) + ... + (x_{n-1}-mu)
    double *d11nn_sum,   // (x0_1-mu)^2 + (x0_n-mu)^2
    double *dtt_sum,     // (x0_2-mu)^2 + ... + (x0_{n-1}-mu)^2
    double *dttp_sum     // (x0_1-mu)(x0_2-mu) + ... + (x0_{n-1}-mu)(x0_n-mu)
);
#endif