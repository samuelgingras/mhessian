#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"


static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name: student_SV_AR\n"
"Description: univariate autoregressive Student't stochastic volatility model, without leverage\n"
"Extra parameters: \n"
"\t nu \t Student's t degree of freedom, positive real scalar\n"
"\t a \t AR(1) constant term\n"
"\t b \t AR(1) correlation coefficient\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_nu = mxGetField(prhs,0,"nu");
    mxArray *pr_a = mxGetField(prhs,0,"a");
    mxArray *pr_b = mxGetField(prhs,0,"b");
    
    ErrMsgTxt( pr_nu != NULL || pr_a != NULL || pr_b != NULL,
        "Invalid input argument: three model paramters expected");
    ErrMsgTxt( mxIsScalar(pr_nu) & mxIsScalar(pr_a) & mxIsScalar(pr_b),
        "Invalid input argument: scalar paramters expected");
    ErrMsgTxt( mxGetScalar(pr_nu) > 0,
        "Invalid input argument: positive argument expected");
    
    theta_y->n = n_theta;
    theta_y->scalar = (double *) mxMalloc(theta_y->n * sizeof( double ));
    theta_y->scalar[0] = mxGetScalar(pr_nu);
    theta_y->scalar[1] = mxGetScalar(pr_a);
    theta_y->scalar[2] = mxGetScalar(pr_b);
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

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int t,n = data->n;
    double nu = theta_y->scalar[0];
    double a = theta_y->scalar[1];
    double b = theta_y->scalar[2];
    
    for (t=0; t<n; t++) {
        double y_tm = (t==0) ? 0.0 : data->y[t-1];
        data->y[t] = a + b * y_tm + rng_t(nu) * exp(alpha[t]/2);
    }
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double nu = theta_y->scalar[0];
    double a = theta_y->scalar[1];
    double b = theta_y->scalar[2];
    double coeff = 0.5 * (nu + 1);
    double result = 0.0;
    
    // Compute log f(y|theta,alpha) 
    for (t=0; t<n; t++) {
        double y_tm = (t==0) ? 0.0 : data->y[t-1];
        double y_t_2 = (data->y[t] - a - b * y_tm) * (data->y[t] - a - b * y_tm);
        result -= coeff * log(1.0 + y_t_2 * exp(-alpha[t]) / nu) + 0.5 * alpha[t];
    }
    *log_f = result + n * (lgamma(coeff) - lgamma(0.5*nu) - 0.5 * log(nu*M_PI));
}

static inline void derivative(double y_t, double alpha_t, double nu, double *psi_t)
{
    double coeff = 0.5 * (nu + 1);
    double x = exp(-alpha_t) * int_pow(y_t,2) / nu;
    double coeff_x = coeff * x;
    double x2 = x * x;
    double x3 = x2 * x;
    double fr1 = 1 / (1 + x);
    double fr2 = fr1 * fr1;
    double fr3 = fr2 * fr1;
    double fr4 = fr3 * fr1;
    double fr5 = fr4 * fr1;
    
    psi_t[1] = coeff_x * fr1 - 0.5;
    psi_t[2] = -coeff_x * fr2;
    psi_t[3] = coeff_x * (1-x) * fr3;
    psi_t[4] = -coeff_x * (1- 4*x + x2) * fr4;
    psi_t[5] = coeff_x * (1- 11*x + 11*x2 - x3) * fr5;
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    double nu = theta->y->scalar[0];
    double a = theta->y->scalar[1];
    double b = theta->y->scalar[2];
    if( t==0 )
        derivative( data->y[t] - a, alpha, nu, psi_t );
    else
        derivative( data->y[t] - a - b * data->y[t-1], alpha, nu, psi_t );
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double nu = theta->y->scalar[0];
    double a = theta->y->scalar[1];
    double b = theta->y->scalar[2];
    double *y = data->y; 
    double *alpha = state->alC; 
    double *psi_t;
    
    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
    {
        double y_tm = (t==0) ? 0.0 : y[t-1];
        derivative( y[t] - a - b*y_tm, alpha[t], nu, psi_t );
    }
}

static void initialize(void);

Observation_model student_SV_AR = { initialize, 0 };

// Fill in record with model specific code
static void initialize()
{
    student_SV_AR.n_partials_t = n_partials_t;
    student_SV_AR.n_partials_tp1 = n_partials_tp1;
    
    student_SV_AR.usage_string = usage_string;
    
    student_SV_AR.initializeParameter = initializeParameter;
    student_SV_AR.read_data = read_data;
    
    student_SV_AR.draw_y__theta_alpha = draw_y__theta_alpha;
    student_SV_AR.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    student_SV_AR.compute_derivatives_t = compute_derivatives_t;
    student_SV_AR.compute_derivatives = compute_derivatives;
}