#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"
#include "model.h"

static int n_theta = 1;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: gammapoisson_SS \n"
"Description: Dynamic Gamma-Poisson count model\n"
"Extra parameters: \n"
"\t r \t Gamma distribution shape parameter, positive real scalar\n";

static int n_dimension_parameters = 0;
static Theta_y_constraints theta_y_constraints[] = {
    {"r", -1, -1, all_positive}
};

static
void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int n = data->n;
    double r = theta_y->matrix[0].p[0];
    
    for(int t=0; t<n; t++)
        data->y[t] = (double) rng_n_binomial( r / (r + exp(x[t])), r );
}

static
void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double r = theta_y->matrix[0].p[0];
    
    *log_f = n * ( r * log(r) - lgamma(r) );

    for (int t=0; t<n; t++)
    {
        *log_f += lgamma(r + data->y[t]) - lgamma(data->y[t] + 1) 
            + data->y[t] * x[t] - (r + data->y[t]) * log(r + exp(x[t]));
    }
}

static inline
void derivative(double y_t, double r, double x_t, double *psi_t)
{
    double x = exp(x_t);
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
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    derivative( data->y[t], theta->y->matrix[0].p[0], x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double r = theta->y->matrix[0].p[0];
    double *k = data->y; 
    double *x = state->alC;	
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride)
        derivative( k[t], r, x[t], psi_t );
}

static
void initializeModel(void);

Observation_model gammapoisson_SS = {"gammapoisson_SS", initializeModel, 0};

static
void initializeModel()
{
    gammapoisson_SS.n_theta = n_theta;
    gammapoisson_SS.n_partials_t = n_partials_t;
    gammapoisson_SS.n_partials_tp1 = n_partials_tp1;
    
    gammapoisson_SS.usage_string = usage_string;
    gammapoisson_SS.theta_y_constraints = theta_y_constraints;
    
    gammapoisson_SS.draw_y__theta_x = draw_y__theta_x;
    gammapoisson_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    gammapoisson_SS.compute_derivatives_t = compute_derivatives_t;
    gammapoisson_SS.compute_derivatives = compute_derivatives;
}