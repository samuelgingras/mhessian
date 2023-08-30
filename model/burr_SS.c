#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"
#include "model.h"

static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: burr_SS \n"
"Description: Burr multiplicative error model\n"
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
    double shape_eta = 1/theta_y->matrix[i_eta].p[0];
    double shape_kappa = 1/theta_y->matrix[i_kappa].p[0];
    double scale_lambda = 1/theta_y->matrix[i_lambda].p[0];
    
    for (int t=0; t<n; t++) {
        double u = rng_rand();
        double v = pow( 1/(1-u), shape_kappa );
        data->y[t] = exp(x[t]) * scale_lambda * pow( v-1, shape_eta );
    }
}

static void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double eta = theta_y->matrix[i_eta].p[0];
    double kappa = theta_y->matrix[i_kappa].p[0];
    double lambda = theta_y->matrix[i_lambda].p[0];

    *log_f = n * ( eta * log(lambda) + log(eta) + log(kappa) );

    for( int t=0; t<n; t++ ) {
        double z = lambda * data->y[t] * exp(-x[t]);
        double z_eta = pow( z, eta );
        *log_f += (eta-1) * log(data->y[t]) - eta * x[t] - (kappa+1) * log1p(z_eta);
    }
}

static inline 
void derivative(double eta, double kappa, double lambda, double y_t, double x_t, double *psi_t)
{   
    double g[6];
    double h[6];
    double q[6];

    // Step 1: Direct computation h(x) = (lambda * y * exp(-x))^eta;
    h[0] = pow( lambda * y_t * exp(-x_t), eta );
    h[1] = -h[0] * eta;
    h[2] = -h[1] * eta;
    h[3] = -h[2] * eta;
    h[4] = -h[3] * eta;
    h[5] = -h[4] * eta;

    // Step 2a: Faa Di Bruno g(x) = q(h(x)) with q(x) = log(1+x);
    double z = 1+h[0];
    double z_inv = 1/z;
    q[0] = log(z);
    q[1] = z_inv;
    q[2] = q[1] * z_inv * (-1.0);
    q[3] = q[2] * z_inv * (-2.0);
    q[4] = q[3] * z_inv * (-3.0);
    q[5] = q[4] * z_inv * (-4.0);
    compute_Faa_di_Bruno( 5, q, h, g );

    // Step 2b: Direct computation of psi(x) = -eta*x - (kappa+1)*g(x)
    psi_t[1] = -(kappa+1) * g[1] - eta;
    psi_t[2] = -(kappa+1) * g[2];
    psi_t[3] = -(kappa+1) * g[3];
    psi_t[4] = -(kappa+1) * g[4];
    psi_t[5] = -(kappa+1) * g[5];

}

static 
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    double eta = theta->y->matrix[i_eta].p[0];
    double kappa = theta->y->matrix[i_kappa].p[0];
    double lambda = theta->y->matrix[i_lambda].p[0];

    derivative( eta, kappa, lambda, data->y[t], x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    double eta = theta->y->matrix[i_eta].p[0];
    double kappa = theta->y->matrix[i_kappa].p[0];
    double lambda = theta->y->matrix[i_lambda].p[0];
    
    int t, n = state->n;
    double *x = state->alC;
    double *psi_t;

    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative( eta, kappa, lambda, data->y[t], x[t], psi_t );
}

static
void initializeModel(void);

Observation_model burr_SS = {"burr_SS", initializeModel, 0};

static
void initializeModel()
{
    burr_SS.n_theta = n_theta;
    burr_SS.n_partials_t = n_partials_t;
    burr_SS.n_partials_tp1 = n_partials_tp1;
    
    burr_SS.usage_string = usage_string;
    burr_SS.theta_y_constraints = theta_y_constraints;
    
    burr_SS.draw_y__theta_x = draw_y__theta_x;
    burr_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    burr_SS.compute_derivatives_t = compute_derivatives_t;
    burr_SS.compute_derivatives = compute_derivatives;
}