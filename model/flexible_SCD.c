#include <math.h>
#include <string.h>

#include <gsl/gsl_poly.h>
#include <gsl/gsl_sf_gamma.h>

#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"

static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: flexible_SCD\n";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_beta = mxGetField( prhs, 0, "beta" );
    mxArray *pr_alpha = mxGetField( prhs, 0, "alpha" );
    mxArray *pr_log_cte = mxGetField( prhs, 0, "log_cte" );
    mxArray *pr_eta = mxGetField( prhs, 0, "eta" );
    mxArray *pr_kappa = mxGetField( prhs, 0, "kappa" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
        
    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_beta);
    theta_y->beta_tm = mxGetDoubles(pr_beta);
    theta_y->eta = mxGetScalar(pr_eta);
    theta_y->kappa = mxGetScalar(pr_kappa);
    theta_y->lambda = mxGetScalar(pr_lambda);

    // Set coefficient for marginal computation
    theta_y->is_data_augmentation = 0;
    if( pr_alpha != NULL )
        theta_y->p_tm = mxGetDoubles(pr_alpha);

    // Set/Compute normalization constants
    if( pr_log_cte != NULL ) {
        theta_y->log_cte_tm = mxGetDoubles(pr_log_cte);
    }
    else {
        theta_y->log_cte_tm = (double *) mxMalloc( (theta_y->m + 1) * sizeof(double) );
        double log_cte = lgamma(theta_y->m + 1);
        for( int j=0; j<theta_y->m; j++ )
            theta_y->log_cte_tm[j] = log_cte - lgamma(j + 1) - lgamma(theta_y->m - j);        
    }

}

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{
    // Check structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input required.");

    // Check nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );
    mxArray *pr_theta_y = mxGetField( prhs, 0, "y" );

    if( pr_theta_x == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Nested structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Nested structure input: Field 'y' required.");

    // Read state and model parameters
    initializeThetaAlpha( pr_theta_x, theta->alpha );
    initializeParameter( pr_theta_y, theta->y );
}

static
void initializeData(const mxArray *prhs, Data *data)
{
    // Check if structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Structure input required.");

    // Set pointer to field
    mxArray *pr_y = mxGetField( prhs, 0, "y" );
    mxArray *pr_s = mxGetField( prhs, 0, "s" );
    
    // Check for missing inputs
    if( pr_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 'y' with observation required.");

    if( pr_s == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 's' with regime index required.");
    
    // Check inputs
    if( !mxIsDouble(pr_y) && mxGetN(pr_y) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( !mxIsDouble(pr_s) && mxGetN(pr_s) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( mxGetM(pr_y) != mxGetM(pr_s) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Incompatible vectors length.");
        
    // Fill in data structure
    data->n = mxGetM(pr_y);                                     // Nb of observation
    data->m = mxGetM(pr_y);                                     // Nb of state
    data->y = mxGetDoubles(pr_y);                               // Observation vector
    data->s = (int *) mxMalloc( data->n * sizeof(int) );        // Component indicator

    // Transform double to int (indicator)
    for( int t=0; t<data->n; t++ )
        data->s[t] = (int) mxGetDoubles(pr_s)[t];
}


static 
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *beta = theta_y->beta_tm;
    double *log_cte = theta_y->log_cte_tm;
    double eta = theta_y->eta;
    double kappa = theta_y->kappa;
    double lambda = theta_y->lambda;
    double eta_kappa = eta * kappa;

    int n = data->n;
    int *s = data->s;
    double *y = data->y;

    *log_f = n * ( log(eta) - lgamma(kappa) + eta_kappa * log(lambda) ); 
    
    for( int t=0; t<n; t++ ) {
        double z_t = pow( lambda * y[t] * exp(-alpha[t]), eta );
        double F_t = gsl_sf_gamma_inc_P( kappa, z_t ); 
        if( theta_y->is_data_augmentation ) {
            *log_f += log(beta[s[t]-1]) + log_cte[s[t]-1];
            *log_f += (s[t]-1) * log(F_t) + (m-s[t]) * log(1-F_t);
        }
        else {
            double F_t_jm1 = 1.0;
            double b_t = 0.0;
            for( int j=0; j<m; j++) {
                b_t += p[j] * F_t_jm1;
                F_t_jm1 = F_t_jm1 * F_t;
            }
            *log_f += log(b_t);
        } 
        *log_f += (eta_kappa - 1) * log(y[t]) - z_t - eta_kappa * alpha[t];
    }
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    // double u = rng_beta(a,b);
    // double y = -log(1-u)/lambda;
}

static inline
void conditional_derivative(
    int m, double eta, double kappa, double lambda, int s_t,
    double y_t, double alpha_t, double *psi_t
    )
{
    double h[6] = { 0.0 };
    double g[6] = { 0.0 };
    double f[6] = { 0.0 };
    double q[6] = { 0.0 };

    double z = y_t * exp(-alpha_t);
    double z_eta = pow(lambda * z, eta);
    double z_inv = 1/z;
    double eta_kappa = eta * kappa;

    // Step 1: Direct computation of central distribution
    double eta2 = eta  * eta;
    double eta3 = eta2 * eta;
    double eta4 = eta3 * eta;
    double eta5 = eta4 * eta;

    psi_t[1] =  eta  * z_eta - eta_kappa;
    psi_t[2] = -eta2 * z_eta;
    psi_t[3] =  eta3 * z_eta;
    psi_t[4] = -eta4 * z_eta;
    psi_t[5] =  eta5 * z_eta;

    // Check if distortion
    if( m > 1 ) {
        // Step 2a: Faa Di Bruno for h(x) = exp(g(x)) with g(x) = -(lambda*x)^eta
        g[0] = -z_eta;
        g[1] = g[0] * z_inv * eta;
        g[2] = g[1] * z_inv * (eta-1);
        g[3] = g[2] * z_inv * (eta-2);
        g[4] = g[3] * z_inv * (eta-3);

        q[0] = q[1] = q[2] = q[3] = q[4] = exp(g[0]);
        compute_Faa_di_Bruno( 4, q, g, h );

        // Step 2b: Leibniz for f(x) = g(x)h(x) with h(x) = cte * x^(eta_kappa-1)
        g[0] = eta * lambda * pow(lambda * z, eta_kappa - 1) / gsl_sf_gamma(kappa);
        g[1] = g[0] * z_inv * (eta_kappa - 1);
        g[2] = g[1] * z_inv * (eta_kappa - 2);
        g[3] = g[2] * z_inv * (eta_kappa - 3);
        g[4] = g[3] * z_inv * (eta_kappa - 4);

        f[0] = gsl_sf_gamma_inc_P( kappa, pow(lambda * z, eta) ); 
        f[1] = g[0]*h[0];
        f[2] = g[1]*h[0] +   g[0]*h[1];
        f[3] = g[2]*h[0] + 2*g[1]*h[1] +   g[0]*h[2]; 
        f[4] = g[3]*h[0] + 3*g[2]*h[1] + 3*g[1]*h[2] +   g[0]*h[3];
        f[5] = g[4]*h[0] + 4*g[3]*h[1] + 6*g[2]*h[2] + 4*g[1]*h[3] + g[0]*h[4];

        // Step 2c: Faa Di Bruno for h(x) = f(g(x)) with g(x) = y * exp(-x)
        g[0] = g[2] = g[4] = z;
        g[1] = g[3] = g[5] = -z;
        compute_Faa_di_Bruno( 5, f, g, h );

        // Step 3: Faa di Bruno for f(x) = log(h(x)) and g(x) = log(1-h(x))
        z = h[0];
        z_inv = 1/z;
        q[0] = log(z);
        q[1] = z_inv;
        q[2] = q[1] * z_inv * (-1.0);
        q[3] = q[2] * z_inv * (-2.0);
        q[4] = q[3] * z_inv * (-3.0);
        q[5] = q[4] * z_inv * (-4.0);
        compute_Faa_di_Bruno( 5, q, h, f );
        
        z = 1 - h[0];
        z_inv = 1/z;
        q[0] = log(z);
        q[1] = -z_inv;
        q[2] = q[1] * z_inv;
        q[3] = q[2] * z_inv * 2.0;
        q[4] = q[3] * z_inv * 3.0;
        q[5] = q[4] * z_inv * 4.0;
        compute_Faa_di_Bruno( 5, q, h, g );
        
        // Step 4: Direct computation to add up derivatives
        for( int d=1; d<6; d++ )
            psi_t[d] += (s_t - 1) * f[d] + (m - s_t) * g[d];
    }
}


static inline
void marginal_derivative(
    int m, double *p, double eta, double kappa, double lambda,
    double y_t, double alpha_t, double *psi_t
    )
{
    double h[6] = { 0.0 };
    double g[6] = { 0.0 };
    double f[6] = { 0.0 };
    double q[6] = { 0.0 };

    double z = y_t * exp(-alpha_t);
    double z_eta = pow(lambda * z, eta);
    double z_inv = 1/z;
    double eta_kappa = eta * kappa;

    // Step 1: Direct computation of central distribution
    double eta2 = eta  * eta;
    double eta3 = eta2 * eta;
    double eta4 = eta3 * eta;
    double eta5 = eta4 * eta;

    psi_t[1] =  eta  * z_eta - eta_kappa;
    psi_t[2] = -eta2 * z_eta;
    psi_t[3] =  eta3 * z_eta;
    psi_t[4] = -eta4 * z_eta;
    psi_t[5] =  eta5 * z_eta;

    if( m > 1 ) {
        // Step 2a: Faa Di Bruno for h(x) = exp(g(x)) with g(x) = -(lambda*x)^eta
        g[0] = -z_eta;
        g[1] = g[0] * z_inv * eta;
        g[2] = g[1] * z_inv * (eta-1);
        g[3] = g[2] * z_inv * (eta-2);
        g[4] = g[3] * z_inv * (eta-3);

        q[0] = q[1] = q[2] = q[3] = q[4] = exp(g[0]);
        compute_Faa_di_Bruno( 4, q, g, h );

        // Step 2b: Leibniz for f(x) = g(x)h(x) with g(x) = cte * x^(eta_kappa-1)
        g[0] = eta * lambda * pow(lambda * z, eta_kappa - 1) / gsl_sf_gamma(kappa);
        g[1] = g[0] * z_inv * (eta_kappa - 1);
        g[2] = g[1] * z_inv * (eta_kappa - 2);
        g[3] = g[2] * z_inv * (eta_kappa - 3);
        g[4] = g[3] * z_inv * (eta_kappa - 4);

        f[0] = gsl_sf_gamma_inc_P( kappa, pow(lambda * z, eta) ); 
        f[1] = g[0]*h[0];
        f[2] = g[1]*h[0] +   g[0]*h[1];
        f[3] = g[2]*h[0] + 2*g[1]*h[1] +   g[0]*h[2]; 
        f[4] = g[3]*h[0] + 3*g[2]*h[1] + 3*g[1]*h[2] +   g[0]*h[3];
        f[5] = g[4]*h[0] + 4*g[3]*h[1] + 6*g[2]*h[2] + 4*g[1]*h[3] + g[0]*h[4];

        // Step 2c: Faa Di Bruno for h(x) = f(g(x)) with g(x) = y * exp(-x)
        g[0] = g[2] = g[4] = z;
        g[1] = g[3] = g[5] = -z;
        compute_Faa_di_Bruno( 5, f, g, h );

        // Step 3: Faa Di Bruno for f(x) = sum_j c_j h(x)^{j-1}
        gsl_poly_eval_derivs( p, m, h[0], q, 6 );
        compute_Faa_di_Bruno( 5, q, h, f );

        // Step 4: Faa di Bruno for h(x) = log(f(x))
        q[0] = log(f[0]);
        q[1] = 1/f[0];
        q[2] = q[1] * q[1] * (-1.0);
        q[3] = q[2] * q[1] * (-2.0);
        q[4] = q[3] * q[1] * (-3.0);
        q[5] = q[4] * q[1] * (-4.0);
        compute_Faa_di_Bruno( 5, q, f, h );

        // Step 5: Direct computation to add up derivatives
        for( int d=1; d<6; d++ )
            psi_t[d] += h[d];
    }
}



static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m =  theta->y->m;
    double *p = theta->y->p_tm;
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;

    if( theta->y->is_data_augmentation )
        conditional_derivative( m, eta, kappa, lambda, data->s[t], data->y[t], alpha, psi_t );
    else
        marginal_derivative( m, p, eta, kappa, lambda, data->y[t], alpha, psi_t );
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;

    int *s = data->s;
    double *y = data->y;

    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;


    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride ) {
        if( theta->y->is_data_augmentation )
            conditional_derivative( m, eta, kappa, lambda, s[t], y[t], alpha[t], psi_t );
        else
            marginal_derivative( m, p, eta, kappa, lambda, y[t], alpha[t], psi_t );
    }
}


static 
void initializeModel(void);

Observation_model flexible_SCD = { initializeModel, 0 };

static 
void initializeModel()
{
    flexible_SCD.n_theta = n_theta;
    flexible_SCD.n_partials_t = n_partials_t;
    flexible_SCD.n_partials_tp1 = n_partials_tp1;
    
    flexible_SCD.usage_string = usage_string;
    
    flexible_SCD.initializeData = initializeData;
    flexible_SCD.initializeTheta = initializeTheta;
    flexible_SCD.initializeParameter = initializeParameter;
    
    flexible_SCD.draw_y__theta_alpha = draw_y__theta_alpha;
    flexible_SCD.log_f_y__theta_alpha = log_f_y__theta_alpha;

    flexible_SCD.compute_derivatives_t = compute_derivatives_t;
    flexible_SCD.compute_derivatives = compute_derivatives;
}
