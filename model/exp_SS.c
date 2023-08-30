#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"


static int n_theta = 0;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name: exp_SS\n"
"Description: Exponential duration model\n"
"Extra parameters: none\n";

static 
void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int t,n = data->n;
    for (t=0; t<n; t++)
        data->y[t] = rng_exp(exp(x[t]));
}

static
void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double result = 0.0;
    for (t=0; t<n; t++)
        result += -x[t] - data->y[t] * exp(-x[t]);
    
    *log_f = result;
}

static inline
void derivative(double y_t, double x_t, double *psi_t)
{
	psi_t[3] = psi_t[5] = y_t * exp(-x_t);
	psi_t[2] = psi_t[4] = -psi_t[3];
	psi_t[1] = psi_t[3] - 1.0;
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
	derivative( data->y[t], x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *x = state->alC;
    
    for( t=0; t<n; t++ )
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], x[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model exp_SS = {"exp_SS", initializeModel, 0};

static
void initializeModel()
{
    exp_SS.n_theta = n_theta;
    exp_SS.n_partials_t = n_partials_t;
    exp_SS.n_partials_tp1 = n_partials_tp1;
    
    exp_SS.usage_string = usage_string;
    exp_SS.theta_y_constraints = NULL;
    
    exp_SS.draw_y__theta_x = draw_y__theta_x;
    exp_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    exp_SS.compute_derivatives_t = compute_derivatives_t;
    exp_SS.compute_derivatives = compute_derivatives;
}