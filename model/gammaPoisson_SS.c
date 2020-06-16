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
"Name: gammaPoisson_SS \n"
"Description: univariate Gamma-Poisson count model\n"
"Extra parameters: \n"
"\t r \t Gamma distribution shape parameter, positive real scalar\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_r = mxGetField(prhs,0,"r");
    
    ErrMsgTxt( pr_r != NULL,
        "Invalid input argument: model parameter expected");
    ErrMsgTxt( mxIsScalar(pr_r),
        "Invalid input argument: scalar parameter expected");
    ErrMsgTxt( mxGetScalar(pr_r) > 0,
        "Invalid input argument: positive parameter expected");
    
    theta_y->n = n_theta;
    theta_y->scalar = (double *) mxMalloc( theta_y->n * sizeof( double ) );
    theta_y->scalar[0] = mxGetScalar(pr_r);
}

static void read_data(const mxArray *prhs, Data *data)
{
    mxArray *pr_y = mxGetField(prhs,0,"y");
    
    ErrMsgTxt( pr_y != NULL,
    "Invalid input argument: data struct: 'y' field missing");
    ErrMsgTxt( mxGetN(pr_y),
    "Invalid input argument: data struct: column vector expected");
    
    data->n = mxGetM(pr_y);
    data->m = mxGetM(pr_y);
    data->y = mxGetPr(pr_y);
}

/*
    The original implementation of the gammaPoisson_SS with E[y_t|x_t] =  r * exp(x_t).
*/

// static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
// {
//     int t,n = data->n;
//     double r = theta_y->scalar[0];
//     for(t=0; t<n; t++)
//         data->y[t] = (double) rng_n_binomial( 1.0 / (1.0 + exp(alpha[t])), r);
// }

// static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
// {
//     int t,n = data->n;
//     double r = theta_y->scalar[0];
//     double result = 0.0;
    
//     // Compute f(y|theta,alpha) with log(y!) = lgamma(y+1)
//     for (t=0; t<n; t++)
//     {
//         result += lgamma(r + data->y[t]) - lgamma(data->y[t] + 1) 
//             + data->y[t] * alpha[t] - (r + data->y[t]) * log(1 + exp(alpha[t]));
//     }
    
//     *log_f = result - n * lgamma(r);
// }

// static inline void derivative(double k_t, double r, double alpha_t, double *psi_t)
// {
//     double x = exp(alpha_t);
//     double coeff_x = -(r + k_t) * x;
//     double x2 = x * x;
//     double x3 = x2 * x;
//     double fr1 = 1 / (1 + x);
//     double fr2 = fr1 * fr1;
//     double fr3 = fr2 * fr1;
//     double fr4 = fr3 * fr1;
//     double fr5 = fr4 * fr1;
    
//     psi_t[1] = coeff_x * fr1 + k_t;
//     psi_t[2] = coeff_x * fr2;
//     psi_t[3] = coeff_x * (1-x) * fr3;
//     psi_t[4] = coeff_x * (1- 4*x + x2) * fr4;
//     psi_t[5] = coeff_x * (1- 11*x + 11*x2 -x3) * fr5;
// }

/*
    The implementation of the gammaPoisson_SS with E[y_t|x_t] = exp(x_t).
*/

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int n = data->n;
    double r = theta_y->scalar[0];
    
    for(int t=0; t<n; t++)
        data->y[t] = (double) rng_n_binomial( r / (r + exp(alpha[t])), r );
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double r = theta_y->scalar[0];
    
    *log_f = n * ( r * log(r) - lgamma(r) );

    for (int t=0; t<n; t++)
    {
        *log_f += lgamma(r + data->y[t]) - lgamma(data->y[t] + 1) 
            + data->y[t] * alpha[t] - (r + data->y[t]) * log(r + exp(alpha[t]));
    }
}

static inline void derivative(double y_t, double r, double alpha_t, double *psi_t)
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

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    derivative( data->y[t], theta->y->scalar[0], alpha, psi_t );
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double r = theta->y->scalar[0];
    double *k = data->y; 
    double *alpha = state->alC;	
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride)
        derivative( k[t], r, alpha[t], psi_t );
}

static void initialize(void);

Observation_model gammaPoisson_SS = { initialize, 0 };

static void initialize()
{
    gammaPoisson_SS.n_partials_t = n_partials_t;
    gammaPoisson_SS.n_partials_tp1 = n_partials_tp1;
    
    gammaPoisson_SS.usage_string = usage_string;
    
    gammaPoisson_SS.initializeParameter = initializeParameter;
    gammaPoisson_SS.read_data = read_data;
    
    gammaPoisson_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    gammaPoisson_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    gammaPoisson_SS.compute_derivatives_t = compute_derivatives_t;
    gammaPoisson_SS.compute_derivatives = compute_derivatives;
}