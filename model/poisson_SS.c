#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"

static int n_theta = 0;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: poisson_SS\n"
"Description: Dynamic Poisson count model\n"
"Extra parameters: none\n";

static 
void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int t,n = data->n;
    for (t=0; t<n; t++)
        data->y[t] = (double) rng_poisson(exp(x[t]));
}

static 
void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double result = 0.0;
    // Compute log f(y|theta,x) with log(y!) = lgamma(y+1)
    for (t=0; t<n; t++)
        result += data->y[t] * x[t] - exp(x[t]) - lgamma(data->y[t]+1);
    
    *log_f = result;
}

static inline
void derivative(double y_t, double x_t, double *psi_t)
{
    psi_t[5] = psi_t[4] = psi_t[3] = psi_t[2] = -exp(x_t);
    psi_t[1] = psi_t[2] + y_t;
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
	derivative(data->y[t], x, psi_t);
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
	int t, n = state->n;
	double *x = state->alC;
  	
  	for( t=0; t<n; t++ ) {
		double *psi_t = state->psi + t * state->psi_stride;
		derivative( data->y[t], x[t], psi_t );
  }
}

static
void initializeModel(void);

Observation_model poisson_SS = {"poisson_SS", initializeModel, 0 };

static
void initializeModel()
{
    poisson_SS.n_theta = n_theta;
    poisson_SS.n_partials_t = n_partials_t;
    poisson_SS.n_partials_tp1 = n_partials_tp1;
    
    poisson_SS.usage_string = usage_string;
    poisson_SS.theta_y_constraints = NULL;
    
    poisson_SS.draw_y__theta_x = draw_y__theta_x;
    poisson_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    poisson_SS.compute_derivatives_t = compute_derivatives_t;
    poisson_SS.compute_derivatives = compute_derivatives;
}