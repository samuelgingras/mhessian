#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"

static int n_theta = 1;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name: gammapoisson_SS \n"
"Description: Dynamic Gamma-Poisson count model\n"
"Extra parameters: \n"
"\t r \t Gamma distribution shape parameter, positive real scalar\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_nu = mxGetField( prhs, 0, "r" );

    // Check for missing parameter
    if( pr_nu == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input: Field 'r' required.");

    // Check parameter
    if( !mxIsScalar(pr_nu) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Model parameter: Scalar parameter required.");

    if( mxGetScalar(pr_nu) < 0.0 )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Model parameter: Positive parameter required.");

    // Read model parameter
    theta_y->r = mxGetScalar(pr_r);
}

static void initializeTheta(const mxArray *prhs, Theta *theta)
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
            "Nested structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Nested structure input: Field 'y' required.");

    // Read state and model parameters
    initializeThetaAlpha( pr_theta_x, theta->alpha );
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

        if( !mxIsDouble(pr_y) || mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(pr_y);
        data->m = mxGetM(pr_y);
        data->y = mxGetDoubles(pr_y);
    }
    else
    {
        if( !mxIsDouble(prhs) || mxGetN(prhs) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int n = data->n;
    double r = theta_y->r;
    
    for(int t=0; t<n; t++)
        data->y[t] = (double) rng_n_binomial( r / (r + exp(alpha[t])), r );
}

static
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double r = theta_y->r;
    
    *log_f = n * ( r * log(r) - lgamma(r) );

    for (int t=0; t<n; t++)
    {
        *log_f += lgamma(r + data->y[t]) - lgamma(data->y[t] + 1) 
            + data->y[t] * alpha[t] - (r + data->y[t]) * log(r + exp(alpha[t]));
    }
}

static inline
void derivative(double y_t, double r, double alpha_t, double *psi_t)
{
    double x = exp(alpha_t);
    double x2 = x * x;
    double x3 = x2 * x;
    double r2 = r * r;
    double r3 = r2 * r;
    double r4 = r3 * r;
    double fr1 = 1 / (r + x);
    double fr2 = fr1 * fr1;
    double fr3 = fr2 * fr1;
    double fr4 = fr3 * fr1;
    double fr5 = fr4 * fr1;
    double coeff_x = -(y_t + r) * x;
    
    psi_t[1] = coeff_x * fr1 + y_t;
    psi_t[2] = coeff_x * fr2 * r;
    psi_t[3] = coeff_x * fr3 * (r2 - r*x);
    psi_t[4] = coeff_x * fr4 * (r3 - 4*r2*x + r*x2);
    psi_t[5] = coeff_x * fr5 * (r4 - 11*r3*x + 11*r2*x2 - r*x3);
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    derivative( data->y[t], theta->y->r, alpha, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double r = theta->y->r;
    double *k = data->y; 
    double *alpha = state->alC;	
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride)
        derivative( k[t], r, alpha[t], psi_t );
}

static
void initializeModel(void);

Observation_model gammapoisson_SS = { initializeModel, 0 };

static
void initializeModel()
{
    gammapoisson_SS.n_theta = n_theta;
    gammapoisson_SS.n_partials_t = n_partials_t;
    gammapoisson_SS.n_partials_tp1 = n_partials_tp1;
    
    gammapoisson_SS.usage_string = usage_string;
    
    gammapoisson_SS.initializeData = initializeData;
    gammapoisson_SS.initializeTheta = initializeTheta;
    gammapoisson_SS.initializeParameter = initializeParameter;
    
    gammapoisson_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    gammapoisson_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    gammapoisson_SS.compute_derivatives_t = compute_derivatives_t;
    gammapoisson_SS.compute_derivatives = compute_derivatives;
}