#include <math.h>
#include <string.h>
#include "RNG.h"
#include "state.h"
#include "model.h"


static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string = 
"Name: mix_gaussian_SV\n"
"Description: Stochastic volatility model, without leverage, with finite mixture of Gaussian\n"
"Extra parameters: for each mixture component j=1,..,J\n"
"\tpi_j\t Component weight of the jth Gaussian distribution\n"
"\tmu_j\t Mean of the jth Gaussian distribution\n"
"\tsigma_j\t Std of the jth Gaussian distribution\n";

static int n_dimension_parameters = 1;
enum {i_p, i_mu, i_sigma, n_th};
static Theta_y_constraints theta_y_constraints[] = {
    {"p", 0, -1, column_stochastic},
    {"mu", 0, -1, NULL},
    {"sigma", 0, -1, all_positive}
};

static
void draw_y__theta_x(double *x, Theta_y *theta_y, Data *data)
{
    int t,j;
    int n = data->n;
    int m = theta_y->dimension_parameters[0];
    double *p = theta_y->matrix[i_p].p;
    double *mu = theta_y->matrix[i_mu].p;
    double *sigma = theta_y->matrix[i_sigma].p;
    double cumul[m];
    
    // Compute cumulative weight
    cumul[0] = p[0];
    for( j=1; j<m; j++ )
        cumul[j] = p[j] + cumul[j-1];
    
    for( t=0; t<n; t++ )
    {
        // Draw component
        int k = 0;
        double u = rng_rand();
        
        while( cumul[k] < u )
            k++;
        
        // Draw data
        data->y[t] = exp(0.5*x[t]) * (mu[k] + sigma[k]*rng_gaussian());
    }
}

static
void log_f_y__theta_x(double *x, Theta_y *theta_y, Data *data, double *log_f)
{
    int t,j;
    int n = data->n;
    int m = theta_y->dimension_parameters[0];
    double *p = theta_y->matrix[i_p].p;
    double *mu = theta_y->matrix[i_mu].p;
    double *sigma = theta_y->matrix[i_sigma].p;
    
    *log_f = -0.5 * n * log(2*M_PI);
    
    for( t=0; t<n; t++ )
    {
        double f_t = 0.0;
        double z_t = data->y[t] * exp(-0.5 * x[t]);
        for( j=0; j<m; j++ )
        {
            double z_t_j = (z_t - mu[j]) / sigma[j];
            f_t += p[j] / sigma[j] * exp(-0.5 * (z_t_j*z_t_j + x[t]));
        }
        *log_f += log(f_t);
    }
}

static inline
void derivative(double y_t, double x_t, int m, double *p, double *mu, double *sigma,
    double *psi_t)
{
    int j,d;
    double g_jt[6];
    double h_jt[6];
    
    double f_t[6];
    double p_t[6] = { 0.0 };
    
    for (j=0; j<m; j++)
    {
        double y_t_j = y_t / sigma[j];
        double mu_sigma_j = mu[j] / sigma[j];

        double A = y_t_j * y_t_j * exp(-x_t);
        double B = y_t_j * mu_sigma_j * exp(-0.5 * x_t);
        double C = mu_sigma_j * mu_sigma_j;
        
        // Step 1 : Direct computation 	
        h_jt[0] = -0.5 * ( A - 2*B + C + x_t );
        h_jt[1] = -0.5 * (-A + B + 1 );
        h_jt[2] = -0.5 * ( A - 0.5*B );
        h_jt[3] = -0.5 * (-A + 0.25*B );
        h_jt[4] = -0.5 * ( A - 0.125*B );
        h_jt[5] = -0.5 * (-A + 0.0625*B );
        
        // Step 2 : Faa di Bruno with g(x) = exp(h(x))
        f_t[0] = f_t[1] = f_t[2] = exp(h_jt[0]);
        f_t[3] = f_t[4] = f_t[5] = exp(h_jt[0]);
        compute_Faa_di_Bruno(5, f_t, h_jt, g_jt);
        
        // Step 3 : Direct computation
        for( d=0; d<6; d++ )
            p_t[d] += p[j] / sigma[j] * g_jt[d];
    }
    
    // Step 4: Faa di Bruno with psi(x) = log(p(x))
    double z = p_t[0];
    double z_2 = z * z;
    double z_3 = z_2 * z;
    double z_4 = z_3 * z;
    double z_5 = z_4 * z;
    
    f_t[0] = log(z);
    f_t[1] =  1.0 / z;
    f_t[2] = -1.0 / z_2;
    f_t[3] =  2.0 / z_3;
    f_t[4] = -6.0 / z_4;
    f_t[5] = 24.0 / z_5;
    
    compute_Faa_di_Bruno(5, f_t, p_t, psi_t);
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    int m = theta->y->dimension_parameters[0];
    double *p = theta->y->matrix[i_p].p;
    double *mu = theta->y->matrix[i_mu].p;
    double *sigma = theta->y->matrix[i_sigma].p;
    
    derivative(data->y[t], x, m, p, mu, sigma, psi_t);
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int m = theta->y->dimension_parameters[0];
    double *p = theta->y->matrix[i_p].p;
    double *mu = theta->y->matrix[i_mu].p;
    double *sigma = theta->y->matrix[i_sigma].p;
    double *x = state->alC; 
    
    double *psi_t;
    int t, n = state->n;
    
    for(t=0, psi_t = state->psi; t < n; t++, psi_t += state->psi_stride )
        derivative(data->y[t], x[t], m, p, mu, sigma, psi_t);
}

static
void initializeModel(void);

Observation_model mix_gaussian_SV = {"mix_gaussian_SV", initializeModel, 0};

static
void initializeModel()
{
    mix_gaussian_SV.n_theta = n_theta;
    mix_gaussian_SV.n_dimension_parameters = n_dimension_parameters;
    mix_gaussian_SV.n_partials_t = n_partials_t;
    mix_gaussian_SV.n_partials_tp1 = n_partials_tp1;
    
    mix_gaussian_SV.usage_string = usage_string;    
    mix_gaussian_SV.theta_y_constraints = theta_y_constraints;
    
    mix_gaussian_SV.draw_y__theta_x = draw_y__theta_x;
    mix_gaussian_SV.log_f_y__theta_x = log_f_y__theta_x;
    
    mix_gaussian_SV.compute_derivatives_t = compute_derivatives_t;
    mix_gaussian_SV.compute_derivatives = compute_derivatives;
}