#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "state.h"
#include "errors.h"


// Default computation options
#define TOLREANCE       0.0001
#define MAX_ITERATIONS     100
#define N_ALPHA_PARTIALS     4

#define TRUE    1
#define FALSE   0

static double *readStateParameter(int n, const mxArray *prhs)
{
    // 
    double *x_tm = (double *) mxCalloc(n, sizeof(double));
    
    if( prhs != NULL ) {
        if(mxIsScalar(prhs)) {
            double x = mxGetScalar(prhs);
            for(int i=0; i<n; i++)
                x_tm[i] = x;
        }
        else {
            mxCheckVector(prhs);
            mxCheckVectorSize(n, prhs);
            memcpy(x_tm, mxGetPr(prhs), n*sizeof(double));
        }
    }
    return x_tm;
}

void initializeThetaAlpha(const mxArray *prhs, State_parameter *theta_alpha)
{
    // Set pointer to each field
    mxArray *pr_N = mxGetField(prhs,0,"N");
    mxArray *pr_d = mxGetField(prhs,0,"d");
    mxArray *pr_mu = mxGetField(prhs,0,"mu");
    mxArray *pr_phi = mxGetField(prhs,0,"phi");
    mxArray *pr_omega = mxGetField(prhs,0,"omega");
    
    // Check for missing field
    ErrMsgTxt( pr_N != NULL,
    "Invalid input argument: number of observation expected");

    // Read number of observation
    ErrMsgTxt( mxIsScalar(pr_N),
    "Invalid input argument: scalar value expected");
    theta_alpha->n = mxGetScalar(pr_N);

    
    // Set default value
    theta_alpha->is_basic = TRUE;
    theta_alpha->is_grad_hess = TRUE;
    

    // Check if model as mean or intercept parameter
    if( pr_d == NULL && pr_mu == NULL ) {
        theta_alpha->alpha_mean = 0.0;
    }

    // Check intercept parameter
    if( pr_d != NULL) {
        if( mxIsScalar(pr_d) ) {
            mexErrMsgIdAndTxt("mhessian:invalidrhs", "Intercept: vector input required.");
        }
        else {
            theta_alpha->is_basic = FALSE;
            theta_alpha->is_grad_hess = FALSE;
        }
    }

    // Check mean parameter
    if( pr_mu != NULL ) {
        if( mxIsScalar(pr_mu) ) {
            theta_alpha->is_mu_basic = TRUE;
            theta_alpha->alpha_mean = mxGetScalar(pr_mu);
        }
        else {
            theta_alpha->is_mu_basic = FALSE;
            theta_alpha->is_basic = FALSE;
        }
    }

    //  Check autocorrelation parameter
    if( pr_phi != NULL ) {
        if( mxIsScalar(pr_phi) ) {
            theta_alpha->is_phi_basic = TRUE;
            theta_alpha->phi = mxGetScalar(pr_phi);
        }
        else {
            theta_alpha->is_phi_basic = FALSE;
            theta_alpha->is_basic = FALSE;
            theta_alpha->is_grad_hess = FALSE;
        }
    }
    else {
        mexErrMsgIdAndTxt("mhessian:invalidrhs", "Autocorrelation parameter required.");
    }

    // Check precision parameter
    if( pr_omega != NULL ) {
        if( mxIsScalar(pr_omega) ) {
            theta_alpha->is_omega_basic = TRUE;
            theta_alpha->omega = mxGetScalar(pr_omega);
        }
        else {
            theta_alpha->is_omega_basic = FALSE;
            theta_alpha->is_basic = FALSE;
            theta_alpha->is_grad_hess = FALSE;
        }
    }
    else {
        mexErrMsgIdAndTxt("mhessian:invalidrhs", "Precision parameter required.");
    }


    // Initialize vector specification
    if( theta_alpha->is_basic ) {
        // Prepare mean vector for joint sampling
        theta_alpha->mu_tm = (double *) mxMalloc( theta_alpha->n * sizeof(double) );
        for(int i=0;i<theta_alpha->n;i++) { theta_alpha->mu_tm[i] = theta_alpha->alpha_mean; }
    }
    else {
        theta_alpha->d_tm = readStateParameter(theta_alpha->n, pr_d);
        theta_alpha->mu_tm = readStateParameter(theta_alpha->n, pr_mu);
        theta_alpha->phi_tm = readStateParameter(theta_alpha->n, pr_phi);
        theta_alpha->omega_tm = readStateParameter(theta_alpha->n, pr_omega);
                
        // Reshape intercept parameters if both mean and intercept are supplied 
        if( pr_mu != NULL ) {
            theta_alpha->d_tm[0] = theta_alpha->d_tm[0] + theta_alpha->mu_tm[0];
            for(int i=1; i<theta_alpha->n; i++) {
                theta_alpha->d_tm[i] = theta_alpha->d_tm[i] + theta_alpha->mu_tm[i] 
                    - theta_alpha->phi_tm[i] * theta_alpha->mu_tm[i-1];
            }
        }

        // Adjust precision of initial state for basic model with mean vector and no intercept
        if( pr_d != NULL && theta_alpha->is_phi_basic && theta_alpha->is_omega_basic ) {
            theta_alpha->omega_tm[0] = 
                theta_alpha->omega * (1 - theta_alpha->phi * theta_alpha->phi);
        }
    }
}
    
State *stateAlloc( void )
{
    State *state = (State *) mxMalloc( sizeof(State) );

    // Set default computation options
    state->guess_alC = FALSE;
    state->sign = TRUE;
    state->tolerance = TOLREANCE;
    state->max_iterations = MAX_ITERATIONS;
    state->max_iterations_safe = 5 * MAX_ITERATIONS;
    state->max_iterations_unsafe = MAX_ITERATIONS;
    state->n_alpha_partials = N_ALPHA_PARTIALS;

    return state;
}

Theta *thetaAlloc( void )
{
    Theta *theta = (Theta *) mxMalloc( sizeof(Theta) );
    theta->y = (Parameter *) mxMalloc( sizeof(Parameter) );
    theta->alpha = (State_parameter *) mxMalloc( sizeof(State_parameter) );
    return theta;
}

Data *dataAlloc( void )
{
    Data *data = (Data *) mxMalloc( sizeof(Data) );
    return data;
}

mxArray *mxStateAlloc(int n, Observation_model *model, State *state)
{
    // List state field
    Field field[] = {
        // Computation variables
        { "x", &(state->alpha) },
        { "Hb_0", &(state->Hb_0) },
        { "Hb_1", &(state->Hb_1) },
        { "cb", &(state->cb) },
        { "Hbb_0", &(state->Hbb_0) },
        { "Hbb_1", &(state->Hbb_1) },
        { "Hbb_1_2", &(state->Hbb_1_2) },
        { "cbb", &(state->cbb) },
        { "xC", &(state->alC) },
        { "Sigma_prior", &(state->Sigma_prior) },
        { "ad_prior", &(state->ad_prior) },
        { "m_prior", &(state->m_prior) },
        { "Sigma", &(state->Sigma) },
        { "m", &(state->m) },
        { "ad", &(state->ad) },
        { "add", &(state->add) },
        { "addd", &(state->addd) },
        { "adddd", &(state->adddd) },
        { "b", &(state->b) },
        { "bd", &(state->bd) },
        { "bdd", &(state->bdd) },
        { "bddd", &(state->bddd) },
        { "mu", &(state->mu) },
        { "mud", &(state->mud) },
        { "mudd", &(state->mudd) },
        { "s", &(state->s) },
        { "sd", &(state->sd) },
        { "sdd", &(state->sdd) },
        { "sddd", &(state->sddd) },
        { "eps", &(state->eps) },
        { "a", &(state->a) },
        { "psi", &(state->psi) }
        // Diagnostic variables
        // ...
    };
    
    // Set state size and fixed options
    state->n = n;
    state->psi_stride = (model->n_partials_t+1) * (model->n_partials_tp1+1);
    
    // Read field names
    int i;
    int nfield = sizeof(field)/sizeof(Field);
    const char *field_names[nfield];
    for(i=0; i<nfield; i++)
        field_names[i] = field[i].Matlab_field_name;
    
    // Prepare matlab structure mxState
    mxArray *field_pr;
    mxArray *mxState = mxCreateStructMatrix(1,1,nfield,field_names);
    for(i=0; i<nfield-1; i++){
        field_pr = mxCreateDoubleMatrix((mwSize)n,1,mxREAL);
        mxSetField(mxState, 0, field[i].Matlab_field_name, field_pr);
        *(field[i].C_field_pointer) = mxGetPr(field_pr);
    }
    field_pr = mxCreateDoubleMatrix((mwSize)state->psi_stride,(mwSize)n,mxREAL);
    mxSetField(mxState, 0, field[i].Matlab_field_name, field_pr);
    *(field[i].C_field_pointer) = mxGetPr(field_pr);
    
    return mxState;
}
