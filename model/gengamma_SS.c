#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"
#include "model.h"

static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: gengamma_SS \n"
"Description: Generalized Gamma multiplicative error model\n"
"Extra parameters: \n"
"\t eta \t \n"
"\t kappa \t "
"\t lambda \t \n";

static int n_dimension_parameters = 0;
enum {i_eta, i_kappa, i_lambda, n_th};
static Theta_y_constraints theta_y_constraints[] = {
    {"eta", -1, -1, all_positive},
    {"kappa", -1, -1, all_positive},
    {"lambda", -1, -1, all_positive}
};

static void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int n = data->n;
    double eta = theta_y->matrix[i_eta].p[0];
    double kappa = theta_y->matrix[i_kappa].p[0];
    double scale = 1/theta_y->matrix[i_lambda].p[0];
    double shape = 1/eta;
    
    for(int t=0; t<n; t++)
    {
        double u = rng_gamma(kappa,1);
        data->y[t] = exp(x[t]) * scale * pow(u,shape);
    }
}

static void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double eta = theta_y->matrix[i_eta].p[0];
    double kappa = theta_y->matrix[i_kappa].p[0];
    double lambda = theta_y->matrix[i_lambda].p[0];
    double eta_kappa = eta * kappa;

    *log_f = n * (log(eta) - lgamma(kappa) + eta_kappa * log(lambda));

    for(int t=0; t<n; t++)
    {
        double y_x_t = data->y[t] * exp(-x[t]) * lambda;
        *log_f += (eta_kappa - 1) * log(data->y[t]) - pow(y_x_t, eta) - eta_kappa * x[t];
    }
}

static inline 
void derivative(double y_t, double eta, double kappa, double lambda, double x_t, double *psi_t)
{   
    double eta2 = eta  * eta;
    double eta3 = eta2 * eta;
    double eta4 = eta3 * eta;
    double eta5 = eta4 * eta;

    double y = y_t * exp(-x_t) * lambda;
    double z = pow(y,eta);
    
    psi_t[1] =  eta  * z - eta * kappa;
    psi_t[2] = -eta2 * z;
    psi_t[3] =  eta3 * z;
    psi_t[4] = -eta4 * z;
    psi_t[5] =  eta5 * z;
}

static 
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    double eta = theta->y->matrix[i_eta].p[0];
    double kappa = theta->y->matrix[i_kappa].p[0];
    double lambda = theta->y->matrix[i_lambda].p[0];

    derivative( data->y[t], eta, kappa, lambda, x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int n = state->n;
    double eta = theta->y->matrix[i_eta].p[0];
    double kappa = theta->y->matrix[i_kappa].p[0];
    double lambda = theta->y->matrix[i_lambda].p[0];
    double *x = state->alC;

    for(int t=0; t<n; t++)
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], eta, kappa, lambda, x[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model gengamma_SS = {"gengamma_SS", initializeModel, 0 };

static
void initializeModel()
{
    gengamma_SS.n_theta = n_theta;
    gengamma_SS.n_partials_t = n_partials_t;
    gengamma_SS.n_partials_tp1 = n_partials_tp1;
    
    gengamma_SS.usage_string = usage_string;
    gengamma_SS.theta_y_constraints = theta_y_constraints;
    
    gengamma_SS.draw_y__theta_x = draw_y__theta_x;
    gengamma_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    gengamma_SS.compute_derivatives_t = compute_derivatives_t;
    gengamma_SS.compute_derivatives = compute_derivatives;
}