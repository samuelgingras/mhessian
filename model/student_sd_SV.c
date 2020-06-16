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
"Name: student_sd_SV\n"
"Description: Univariate normalized Student's t stochastic volatility model, without leverage\n"
"Extra parameters:\n"
"\tnu\tStudent's t degree of freedom, positive real scalar\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    theta_y->n = n_theta;
    mxArray *pr_nu = mxGetField(prhs,0,"nu");
    
    ErrMsgTxt( pr_nu != NULL,
        "Invalid input argument: model parameter expected");
    ErrMsgTxt( mxIsScalar(pr_nu),
        "Invalid input argument: scalar parameter expected");
    ErrMsgTxt( mxGetScalar(pr_nu) > 0,
        "Invalid input argument: positive parameter expected");
    
    theta_y->scalar = (double *) mxMalloc( sizeof(double) );
    theta_y->scalar[0] = mxGetScalar(pr_nu);
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
    for (t=0; t<n; t++)
        data->y[t] = sqrt((nu-2) / nu) * rng_t(nu) * exp(alpha[t]/2);
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double nu = theta_y->scalar[0];
    double coeff = 0.5 * (nu + 1);
    double result = 0.0;
    for (t=0; t<n; t++) {
        double y_t_2 = int_pow(data->y[t],2);
        result -= coeff * log( 1.0 + y_t_2 * exp(-alpha[t]) / (nu-2) ) + 0.5 * alpha[t];
    }
    
    result += n * ( lgamma(coeff) - lgamma(0.5*nu) );
    result -= 0.5 * n * ( log((nu-2)*M_PI) );

    *log_f = result;
}

static inline void derivative( double y_t, double alpha_t, double nu, double *psi_t )
{
    double coeff = 0.5 * (nu + 1);
    double x = exp(-alpha_t) * int_pow(y_t,2) / (nu-2);
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
    derivative(data->y[t], alpha, theta->y->scalar[0], psi_t);
}

static void compute_derivatives( Theta *theta, State *state, Data *data )
{
    int t, n = state->n;
    double nu = theta->y->scalar[0];
    double *y = data->y; 
    double *alpha = state->alC; 
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride) {
        derivative( y[t], alpha[t], nu, psi_t );
  	}
}

static void initialize(void);

Observation_model student_sd_SV = { initialize, 0 };

static void initialize()
{
    student_sd_SV.n_partials_t = n_partials_t;
    student_sd_SV.n_partials_tp1 = n_partials_tp1;
    
    student_sd_SV.usage_string = usage_string;
    
    student_sd_SV.initializeParameter = initializeParameter;
    student_sd_SV.read_data = read_data;
    
    student_sd_SV.draw_y__theta_alpha = draw_y__theta_alpha;
    student_sd_SV.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    student_sd_SV.compute_derivatives_t = compute_derivatives_t;
    student_sd_SV.compute_derivatives = compute_derivatives;
}