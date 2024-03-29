#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"
#include "model.h"

static int n_theta = 1;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: gamma_SS \n"
"Description: Gamma multiplicative error model\n"
"Extra parameters: \n"
"\t kappa \t \n";

static int n_dimension_parameters = 0;
static Theta_y_constraints theta_y_constraints[] = {
    {"kappa", -1, -1, all_positive}
};

static void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int n = data->n;
    double kappa = theta_y->matrix[0].p[0];
    double scale = 1/kappa;
    
    for( int t=0; t<n; t++ )
        data->y[t] = exp(x[t]) * scale * rng_gamma(kappa,1);
}

static void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double kappa = theta_y->matrix[0].p[0];

    *log_f = n * (kappa * log(kappa) - lgamma(kappa));

    for(int t=0; t<n; t++) {
        double y_x_t = data->y[t] * exp(-x[t]) * kappa;
        *log_f += (kappa - 1) * log(data->y[t]) - kappa * x[t] - y_x_t;
    }
}

static inline void derivative(double y_t, double kappa, double x_t, double *psi_t)
{   
    psi_t[3] = psi_t[5] = y_t * exp(-x_t) * kappa;
    psi_t[2] = psi_t[4] = -psi_t[3];
    psi_t[1] = psi_t[3] - kappa;
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    double kappa = theta->y->matrix[0].p[0];
    derivative(data->y[t], kappa, x, psi_t);
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int n = state->n;
    double *x = state->alC;
    double kappa = theta->y->matrix[0].p[0];

    for(int t=0; t<n; t++) {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], kappa, x[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model gamma_SS = {"gamma_SS", initializeModel, 0};

static
void initializeModel()
{
    gamma_SS.n_theta = n_theta;
    gamma_SS.n_partials_t = n_partials_t;
    gamma_SS.n_partials_tp1 = n_partials_tp1;
    
    gamma_SS.usage_string = usage_string;
    gamma_SS.theta_y_constraints = theta_y_constraints;
    
    gamma_SS.draw_y__theta_x = draw_y__theta_x;
    gamma_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    gamma_SS.compute_derivatives_t = compute_derivatives_t;
    gamma_SS.compute_derivatives = compute_derivatives;
}