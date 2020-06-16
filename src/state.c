#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "state.h"
#include "errors.h"


// Default computation options
static double tolerance = 0.0001;
static int max_iterations = 100;
static int n_alpha_partials = 4;


static double *readStateParameter(int n, const mxArray *prhs)
{
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

void initializeStateParameter(const mxArray *prhs, State_parameter *theta_alpha)
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
    ErrMsgTxt( pr_d != NULL || pr_mu != NULL,
    "Invalid input argument: mean/intercept parameter expected");
    ErrMsgTxt( pr_phi != NULL,
    "Invalid input argument: persitence parameter expected");
    ErrMsgTxt( pr_omega != NULL,
    "Inbalid input argument: precision parameter expected");
    
    // Read number of observation
    ErrMsgTxt( mxIsScalar(pr_N),
    "Invalid input argument: scalar value expected");
    theta_alpha->n = mxGetScalar(pr_N);
    
    // Check if basic specification
    theta_alpha->is_basic = (mxIsScalar(pr_mu) & mxIsScalar(pr_phi) & mxIsScalar(pr_omega));
    
    // Read parameters
    if( theta_alpha->is_basic ) {
        theta_alpha->alpha_mean = mxGetScalar(pr_mu);
        theta_alpha->phi = mxGetScalar(pr_phi);
        theta_alpha->omega = mxGetScalar(pr_omega);
    }
    else {
        theta_alpha->d_tm = readStateParameter(theta_alpha->n, pr_d);
        theta_alpha->mu_tm = readStateParameter(theta_alpha->n, pr_mu);
        theta_alpha->phi_tm = readStateParameter(theta_alpha->n, pr_phi);
        theta_alpha->omega_tm = readStateParameter(theta_alpha->n, pr_omega);
        
        
        double *d_tm = theta_alpha->d_tm;
        double *mu_tm = theta_alpha->mu_tm;
        double *phi_tm = theta_alpha->phi_tm;
        double *omega_tm = theta_alpha->omega_tm;
        
        if( pr_mu != NULL ) {
            d_tm[0] = d_tm[0] + mu_tm[0];
            for(int i=1; i<theta_alpha->n; i++)
                d_tm[i] = d_tm[i] + mu_tm[i] - phi_tm[i] * mu_tm[i-1];
        }
        
        if( mxIsScalar(pr_omega) ) {
            ErrMsgTxt( mxIsScalar(pr_phi),
            "Invalid input argument: scalar value expected");
            omega_tm[0] = omega_tm[0] * (1 - phi_tm[0] * phi_tm[0]);
            
            if( pr_mu == NULL )
                d_tm[0] = d_tm[0] / (1 - phi_tm[0]);
        }
    }
}

void setDefaultOptions(State *state)
{
    state->guess_alC = 0;
    state->sign = 1;
    state->tolerance = tolerance;
    state->max_iterations = max_iterations;
    state->max_iterations_safe = 5 * max_iterations;
    state->max_iterations_unsafe = max_iterations;
    state->n_alpha_partials = n_alpha_partials;
}

void readComputationOptions(const mxArray *prhs, State *state)
{
    // Set pointer to field options
    mxArray *pr_tol = mxGetField(prhs,0,"tolerance");
    mxArray *pr_max = mxGetField(prhs,0,"max_iterations");
    mxArray *pr_par = mxGetField(prhs,0,"n_x_partials");
    mxArray *pr_alc = mxGetField(prhs,0,"x_mode");
    
    // Check tolerance option
    if( pr_tol != NULL ) {
        ErrMsgTxt( mxIsScalar(pr_tol),
            "Invalid option: scalar argument expected");
        state->tolerance = mxGetScalar(pr_tol);
    }
    
    // Check max iterations option
    if( pr_max != NULL ) {
        ErrMsgTxt( mxIsScalar(pr_max),
            "Invalid option: scalar argument expected");
        int max = mxGetScalar(pr_max);
        state->max_iterations = max;
        state->max_iterations_safe = 5 * max;
        state->max_iterations_unsafe = max;
    }
    
    // Check partial derivative option
    if( pr_par != NULL ) {
        ErrMsgTxt( mxIsScalar(pr_par),
            "Invalid option: scalar argument expected");
        state->n_alpha_partials = mxGetScalar(pr_par);
    }
    
    // Check mode guess option
    if( pr_alc != NULL ) {
        state->guess_alC = 1;
        mxCheckVector(pr_alc);
        mxCheckVectorSize(state->n, pr_alc);
        memcpy(state->alC, mxGetPr(pr_alc), state->n * sizeof(double));
    }
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
        { "x_mode", &(state->alC) },
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
    int i, nfield = sizeof(field)/sizeof(Field);
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

Theta *mxThetaAlloc( void )
{
    Theta *theta = (Theta *) mxMalloc( sizeof(Theta) );
    theta->y = (Parameter *) mxMalloc( sizeof(Parameter) );
    theta->alpha = (State_parameter *) mxMalloc( sizeof(State_parameter) );
    return theta;
}