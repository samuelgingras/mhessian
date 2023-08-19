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
"Name: mix_gamma_SS \n"
"Description: Mixture of gamma multiplicative error model\n"
"Extra parameters: for j=1,...,J\n"
"\tp_j\t Component weight of the jth exponential distribution\n"
"\tkappa_j\t Shape parameter of the jth component"
"\tlambda_j\t Scale parameter of the jth component";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_p = mxGetField( prhs, 0, "p" );
    mxArray *pr_kappa = mxGetField( prhs, 0, "kappa" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );

    // Check for missing parameter
    if( pr_p == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'p' required.");

    if( pr_kappa == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'kappa' required.");

    if( pr_lambda == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'lambda' required.");
    
    // Check parameters
    if( !mxIsDouble(pr_p) || mxGetN(pr_p) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_kappa) || mxGetN(pr_kappa) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_lambda) || mxGetN(pr_lambda) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( mxGetM(pr_p) != mxGetM(pr_lambda) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    if( mxGetM(pr_p) != mxGetM(pr_kappa) )
    mexErrMsgIdAndTxt( "mhessian:invalidInputs",
        "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_p);
    theta_y->p_tm = mxGetDoubles(pr_p);
    theta_y->kappa_tm = mxGetDoubles(pr_kappa);
    theta_y->lambda_tm = mxGetDoubles(pr_lambda);

    // Compute normalization constant of each component
    theta_y->cte_tm = (double *) mxMalloc( theta_y->m * sizeof(double) );
    double log_cte_j;
    for( int j=0; j<theta_y->m; j++ ) {
        log_cte_j = theta_y->kappa_tm[j] * log(theta_y->lambda_tm[j]);
        log_cte_j = log_cte_j - lgamma(theta_y->kappa_tm[j]);
        theta_y->cte_tm[j] = exp(log_cte_j);
    }
}

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{

    // Check structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input required.");

    // Check nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );
    mxArray *pr_theta_y = mxGetField( prhs, 0, "y" );

    if( pr_theta_x == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input: Field 'y' required.");

    // Read state and model parameters
    initializeThetax( pr_theta_x, theta->x );
    initializeParameter( pr_theta_y, theta->y );

}

static
void initializeData(const mxArray *prhs, Data *data)
{
    if( mxIsStruct(prhs) )
    {
        mxArray *pr_y = mxGetField( prhs, 0, "y" );

        if( pr_y == NULL )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:missingInputs",
                "Structure input: Field 'y' required.");

        if( !mxIsDouble(pr_y) )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Vector of double required.");

        if( mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector required.");

        data->n = mxGetM(pr_y);
        data->m = mxGetM(pr_y);
        data->y = mxGetDoubles(pr_y);
    }
    else
    {
        if( !mxIsDouble(prhs) && mxGetN(prhs) != 1 )
            mexErrMsgIdAndTxt( "mhessian:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static void draw_y__theta_x(double *x, Parameter *theta_y, Data *data)
{
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *kappa = theta_y->kappa_tm;
    double *lambda = theta_y->lambda_tm;

    // Pre-computation
    double cumul[m];
    cumul[0] = p[0];
    for( int j=1; j<m; j++ )
        cumul[j] = cumul[j-1] + p[j];

    for( int t=0; t<n; t++ ) {
        // Draw mixture component
        int k=0;
        double u = rng_rand();
        while( cumul[k] < u )
            k++;

        // Draw observation
        data->y[t] = exp(x[t]) / lambda[k] * rng_gamma(kappa[k],1);
    }
}

static double log_f_y__theta_x_t(
    int m,
    double *cte,
    double *p, 
    double *kappa, 
    double *lambda,
    double y_t,
    double x_t
    )
{
    double result = 0.0;
    for( int j=0; j<m; j++ ) {
        double y_jt = lambda[j] * y_t * exp(-x_t);
        double f_y_jt = pow(y_t, kappa[j]-1) * exp(-y_jt - kappa[j] * x_t);
        result += p[j] * cte[j] * f_y_jt;
    }
    return log(result);
}

static void log_f_y__theta_x(double *x, Parameter *theta_y, Data *data, double *log_f)
{
    int m = theta_y->m;
    double *cte = theta_y->cte_tm;
    double *p = theta_y->p_tm;
    double *kappa = theta_y->kappa_tm;
    double *lambda = theta_y->lambda_tm;

    *log_f = 0.0;
    for(int t=0; t<data->n; t++)
        *log_f += log_f_y__theta_x_t( m, cte, p, kappa, lambda, data->y[t], x[t] );
}

static inline void derivative(
    int m,
    double *cte,
    double *p,
    double *kappa,
    double *lambda,
    double y_t,
    double x_t,
    double *psi_t
    )
{   
    double h_jt[6];
    double g_jt[6];
    
    double q[6];                // Composition function for Faa di Bruno
    double f_t[6] = {0.0};      // Evaluation of f(y_t|x_t) and derivatives
    

    for( int j=0; j<m; j++ ) {

        // Step 1: Direct computation
        h_jt[3] = h_jt[5] = lambda[j] * y_t * exp(-x_t);
        h_jt[2] = h_jt[4] = -h_jt[3];
        h_jt[1] = h_jt[3] - kappa[j];
        h_jt[0] = h_jt[2] - kappa[j]*x_t;

        // Step 2: Faa di Bruno with g(x) = exp(h(x))
        q[0]= exp(h_jt[0]);
        q[1]= exp(h_jt[0]);
        q[2]= exp(h_jt[0]);
        q[3]= exp(h_jt[0]);
        q[4]= exp(h_jt[0]);
        q[5]= exp(h_jt[0]);
        compute_Faa_di_Bruno(5, q, h_jt, g_jt);

        // Step 3: Direct computation
        for( int d=0; d<6; d++ )
            f_t[d] += p[j] * cte[j] * pow(y_t, kappa[j]-1) * g_jt[d];
    }

    // Step 4: Faa di Bruno with psi(x) = log(f(x))
    double z = f_t[0];
    double z_2 = z * z;
    double z_3 = z * z_2;
    double z_4 = z * z_3;
    double z_5 = z * z_4;

    q[0] = log(z);
    q[1] =  1.0 / z;
    q[2] = -1.0 / z_2;
    q[3] =  2.0 / z_3;
    q[4] = -6.0 / z_4;
    q[5] = 24.0 / z_5;
    compute_Faa_di_Bruno(5, q, f_t, psi_t);
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    int m = theta->y->m;
    double *cte = theta->y->cte_tm;
    double *p = theta->y->p_tm;
    double *kappa = theta->y->kappa_tm;
    double *lambda = theta->y->lambda_tm;

    derivative( m, cte, p, kappa, lambda, data->y[t], x, psi_t );
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{

    int n = state->n;
    double *y = data->y;
    double *x = state->alC;

    int m =  theta->y->m;
    double *cte = theta->y->cte_tm;
    double *p = theta->y->p_tm;
    double *kappa = theta->y->kappa_tm;
    double *lambda = theta->y->lambda_tm;

    for( int t=0; t<n; t++ ) {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( m, cte, p, kappa, lambda, y[t], x[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model mix_gamma_SS = { initializeModel, 0 };

static
void initializeModel()
{
    mix_gamma_SS.n_theta = n_theta;
    mix_gamma_SS.n_partials_t = n_partials_t;
    mix_gamma_SS.n_partials_tp1 = n_partials_tp1;
    
    mix_gamma_SS.usage_string = usage_string;
    
    mix_gamma_SS.initializeData = initializeData;
    mix_gamma_SS.initializeTheta = initializeTheta;
    mix_gamma_SS.initializeParameter = initializeParameter;
    
    mix_gamma_SS.draw_y__theta_x = draw_y__theta_x;
    mix_gamma_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    mix_gamma_SS.compute_derivatives_t = compute_derivatives_t;
    mix_gamma_SS.compute_derivatives = compute_derivatives;
}