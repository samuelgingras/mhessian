#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"

static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: flexible_SCD\n";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_p = mxGetField( prhs, 0, "beta" );
    mxArray *pr_eta = mxGetField( prhs, 0, "eta" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
    
    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_p);
    theta_y->p_tm = mxGetDoubles(pr_p);
    theta_y->eta = mxGetScalar(pr_eta);
    theta_y->lambda = mxGetScalar(pr_lambda);

    // Compute normalization constants
    theta_y->log_cte_tm = (double *) mxMalloc( (theta_y->m + 1) * sizeof(double) );
    double log_cte = lgamma(theta_y->m + 1);
    for( int j=0; j<theta_y->m; j++ )
        theta_y->log_cte_tm[j] = log_cte - lgamma(j + 1) - lgamma(theta_y->m - j);
}

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{
    // Check structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input required.");

    // Check nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );
    mxArray *pr_theta_y = mxGetField( prhs, 0, "y" );

    if( pr_theta_x == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Nested structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Nested structure input: Field 'y' required.");

    // Read state and model parameters
    initializeThetaAlpha( pr_theta_x, theta->alpha );
    initializeParameter( pr_theta_y, theta->y );
}

static
void initializeData(const mxArray *prhs, Data *data)
{
    // Check if structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Structure input required.");

    // Set pointer to field
    mxArray *pr_y = mxGetField( prhs, 0, "y" );
    mxArray *pr_s = mxGetField( prhs, 0, "s" );
    
    // Check for missing inputs
    if( pr_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 'y' with observation required.");

    if( pr_s == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 's' with regime index required.");
    
    // Check inputs
    if( !mxIsDouble(pr_y) && mxGetN(pr_y) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( !mxIsDouble(pr_s) && mxGetN(pr_s) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( mxGetM(pr_y) != mxGetM(pr_s) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Incompatible vectors length.");
        
    // Fill in data structure
    data->n = mxGetM(pr_y);                                     // Nb of observation
    data->m = mxGetM(pr_y);                                     // Nb of state
    data->y = mxGetDoubles(pr_y);                               // Observation vector
    data->s = (int *) mxMalloc( data->n * sizeof(int) );        // Component indicator

    // Transform double to int (indicator)
    for( int t=0; t<data->n; t++ )
        data->s[t] = (int) mxGetDoubles(pr_s)[t];
}

static 
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int m = theta_y->m;
    double *beta = theta_y->p_tm;
    double *log_cte = theta_y->log_cte_tm;
    double eta = theta_y->eta;
    double lambda = theta_y->lambda;
    
    int n = data->n;
    int *s = data->s;
    double *y = data->y;

    *log_f = n * ( log(eta) + eta * log(lambda) );

    for( int t=0; t<n; t++ ) {
        int k = s[t]-1;
        
        double g_t = -pow( lambda * y[t] * exp(-alpha[t]), eta );
        double f_t = log(1 - exp(g_t));

        *log_f += (eta-1) * log(y[t]) - eta * alpha[t];
        *log_f += log(beta[k]) + log_cte[k] + (s[t]-1) * f_t + (m-s[t]+1) * g_t;
    }
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    // double u = rng_beta(a,b);
    // double y = -log(1-u)/lambda;
}

static inline
void derivative(
    int m, double eta, double lambda, 
    int s_t, double y_t, double alpha_t,
    double *psi_t
    )
{
        
    double h[6] = { 0.0 };
    double g[6] = { 0.0 };
    double f[6] = { 0.0 };
    double q[6] = { 0.0 };

    // Step 1: Direct computation of g(x) = -(lambda*y*exp(-x))^eta
    g[0] = -pow(lambda * y_t * exp(-alpha_t), eta);
    g[1] = -g[0] * eta;
    g[2] = -g[1] * eta;
    g[3] = -g[2] * eta;
    g[4] = -g[3] * eta;
    g[5] = -g[4] * eta;

    // Step 2: Faa di Bruno for h(x) = 1-exp(g(x))
    q[0] = 1-exp(g[0]);
    q[1] = q[0]-1;
    q[2] = q[1];
    q[3] = q[1];
    q[4] = q[1];
    q[5] = q[1];
    compute_Faa_di_Bruno( 5, q, g, h );

    // Step 3: Faa di Bruno with f(x) = log(h(x));
    q[0] = log(h[0]);
    q[1] = 1/h[0];
    q[2] = q[1] * q[1] * (-1.0);
    q[3] = q[2] * q[1] * (-2.0);
    q[4] = q[3] * q[1] * (-3.0);
    q[5] = q[4] * q[1] * (-4.0);
    compute_Faa_di_Bruno( 5, q, h, f );
    
    // Step 4: Direct computation to add up derivatives
    double a = s_t - 1;
    double b = m - s_t + 1;

    psi_t[1] = a * f[1] + b * g[1] - eta;
    psi_t[2] = a * f[2] + b * g[2];
    psi_t[3] = a * f[3] + b * g[3];
    psi_t[4] = a * f[4] + b * g[4];
    psi_t[5] = a * f[5] + b * g[5];
}


static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m =  theta->y->m;
    double eta = theta->y->eta;
    double lambda = theta->y->lambda;

    derivative( m, eta, lambda, data->s[t], data->y[t], alpha, psi_t );
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    
    int m = theta->y->m;
    double eta = theta->y->eta;
    double lambda = theta->y->lambda;

    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;

    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative( m, eta, lambda, data->s[t], data->y[t], alpha[t], psi_t );
}


static 
void initializeModel(void);

Observation_model flexible_SCD = { initializeModel, 0 };

static 
void initializeModel()
{
    flexible_SCD.n_theta = n_theta;
    flexible_SCD.n_partials_t = n_partials_t;
    flexible_SCD.n_partials_tp1 = n_partials_tp1;
    
    flexible_SCD.usage_string = usage_string;
    
    flexible_SCD.initializeData = initializeData;
    flexible_SCD.initializeTheta = initializeTheta;
    flexible_SCD.initializeParameter = initializeParameter;
    
    flexible_SCD.draw_y__theta_alpha = draw_y__theta_alpha;
    flexible_SCD.log_f_y__theta_alpha = log_f_y__theta_alpha;

    flexible_SCD.compute_derivatives_t = compute_derivatives_t;
    flexible_SCD.compute_derivatives = compute_derivatives;
}
