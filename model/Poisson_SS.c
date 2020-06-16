#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"


static int n_theta = 0;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: Poisson_SS\n"
"Description: univariate Poisson count model\n"
"Extra parameters: none\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    theta_y->n = n_theta;
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
    for (t=0; t<n; t++)
        data->y[t] = (double) rng_poisson(exp(alpha[t]));
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double result = 0.0;
    // Compute log f(y|theta,alpha) with log(y!) = lgamma(y+1)
    for (t=0; t<n; t++)
        result += data->y[t] * alpha[t] - exp(alpha[t]) - lgamma(data->y[t]+1);
    
    *log_f = result;
}

static inline void derivative(double y_t, double alpha_t, double *psi_t)
{
    psi_t[5] = psi_t[4] = psi_t[3] = psi_t[2] = -exp(alpha_t);
    psi_t[1] = psi_t[2] + y_t;
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
	derivative(data->y[t], alpha, psi_t);
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
	int t, n = state->n;
	double *alpha = state->alC;
  	
  	for( t=0; t<n; t++ ) {
		double *psi_t = state->psi + t * state->psi_stride;
		derivative( data->y[t], alpha[t], psi_t );
  }
}

static void initialize(void);

Observation_model Poisson_SS = { initialize, 0 };

static void initialize()
{
    Poisson_SS.n_partials_t = n_partials_t;
    Poisson_SS.n_partials_tp1 = n_partials_tp1;
    
    Poisson_SS.usage_string = usage_string;
    
    Poisson_SS.initializeParameter = initializeParameter;
    Poisson_SS.read_data = read_data;
    
    Poisson_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    Poisson_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    Poisson_SS.compute_derivatives_t = compute_derivatives_t;
    Poisson_SS.compute_derivatives = compute_derivatives;
}