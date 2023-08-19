#include "mex.h"

#ifndef MEX_STATE
#define MEX_STATE

typedef struct {
    int n;              // Nb of observation
    int m;              // Nb of state
    double *y;          // Vector of observations

    int is_index;       // For data augmentation in mixture models
    int *s;             // Regime indicator
    int *k;             // State indicator
    int *p;             // Position of state (computed using state indicator)
} Data;

typedef struct {
    int n;              // Nb of scalar
    double *scalar;

    int m;              // Nb of mixture
    
    double r;
    double nu;
    double x;
    double beta;
    double eta;
    double kappa;
    double lambda;

    int is_data_augmentation;
    
    double *p_tm;
    double *beta_tm;
    double *mu_tm;
    double *sigma_tm;
    double *eta_tm;
    double *kappa_tm;
    double *lambda_tm;

    double *cte_tm;
    double *log_cte_tm;

    double *q_tm;
    int *k_tm;
    
} Parameter;

typedef struct {
    char *name;
    int row_dimension_index;
    int col_dimension_index;
    int (*verify)(int n_elements, double *p);
} Parameter_specification;

typedef struct {
    int n;
    int is_basic;

    // For gradHess approximation
    int is_mu_basic;
    int is_phi_basic;
    int is_omega_basic;
    int is_grad_hess;
    
    double x_mean;
    double phi;
    double omega;
    
    double *d_tm;
    double *mu_tm;
    double *phi_tm;
    double *omega_tm;
} State_parameter;

typedef struct {
    Parameter *y;
    State_parameter *x;
} Theta;

typedef struct {
    int n;
    int sign;
    int psi_stride;
    
    // Computation options
    double tolerance;
    int max_iterations;                 // WJM: delete if not using
    int max_iterations_safe;
    int max_iterations_unsafe;
    int n_x_partials;
    
    int circ_buffer_pos;                // SG: not used
    int circ_buffer_len;                // SG: not used
    double *inf_norm_circ_buffer;       // SG: not used
    
    
    int iteration;                      // SG: not used
    int guess_alC;                      // SG: for initialization
    int trust_alC;                      // SG: remove as input for compute_alC (use state)
    int safe;                           // SG: remove as input for compute_alC (use state)
    
    // Computation variables
    double *x;
    double *Hb_0;
    double *Hb_1;
    double *cb;
    double *Hbb_0;
    double *Hbb_1;
    double *Hbb_1_2;
    double *cbb;
    double *alC;
    double *Sigma_prior;
    double *ad_prior;
    double *m_prior;
    double *Sigma;
    double *m;
    double *ad;
    double *add;
    double *addd;
    double *adddd;
    double *b;
    double *bd;
    double *bdd;
    double *bddd;
    double *mu;
    double *mud;
    double *mudd;
    double *s;
    double *sd;
    double *sdd;
    double *sddd;
    double *eps;
    double *a;
    double *psi;
    
    // Diagnostic variables
    // Documented in compute_diagnostics function in x_univariate.c
    int compute_diagnostics;
    int fatal_error_detected;
    double *psi2ratio;
    double *h3norm;
    double *h4norm;
    double *h5norm;
    double *psi3norm;
    double *psi4norm;
    double *psi5norm;
    double *s2priornorm;
} State;

typedef struct {
    void (*initializeModel)(void);
    void (*initializeData)(const mxArray *prhs, Data *data);    
    void (*initializeTheta)(const mxArray *prhs, Theta *theta);
    void (*initializeParameter)(const mxArray *prhs, Parameter *theta_y);
    
    void (*draw_y__theta_x)(double *x, Parameter *theta_y, Data *data);
    void (*log_f_y__theta_x)(double *x, Parameter *theta_y, Data *data, double *log_f);
    
    void (*compute_derivatives_t)(Theta *theta, Data *data, int t, double x, double *psi_t);
    void (*compute_derivatives)(Theta *theta, State *state, Data *data);
    
    char *usage_string;
    int n_theta;
    int n_partials_t;
    int n_partials_tp1;
} Observation_model;

typedef struct {
    char *Matlab_field_name;
    double **C_field_pointer;
} Field;

void initializeThetax( const mxArray *prhs, State_parameter *theta_x );
State *stateAlloc( void );
Theta *thetaAlloc( void );
Data *dataAlloc( void );
mxArray *mxStateAlloc( int n, Observation_model *model, State *state );
#endif