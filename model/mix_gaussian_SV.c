#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"


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

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_p = mxGetField( prhs, 0, "p" );
    mxArray *pr_mu = mxGetField( prhs, 0, "mu" );
    mxArray *pr_sigma = mxGetField( prhs, 0, "sigma" );
    
    // Check for missing parameters
    if( pr_p == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'p' required.");

    if( pr_mu == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'mu' required.");

    if( pr_sigma == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'sigma' required.");

    // Check parameters
    if( !mxIsDouble(pr_p) || mxGetN(pr_p) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_mu) || mxGetN(pr_mu) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( !mxIsDouble(pr_sigma) || mxGetN(pr_sigma) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( mxGetM(pr_p) != mxGetM(pr_mu) || mxGetM(pr_p) != mxGetM(pr_sigma) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_p);
    theta_y->p_tm = mxGetDoubles(pr_p);
    theta_y->mu_tm = mxGetDoubles(pr_mu);
    theta_y->sigma_tm = mxGetDoubles(pr_sigma);
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
    initializeThetax( pr_theta_x, theta->x );
    initializeParameter( pr_theta_y, theta->y );
}

static
void initializeData(const mxArray *prhs, Data *data)
{
    if( mxIsStruct(prhs) )
    {
        mxArray *pr_y = mxGetField( prhs, 0, "y" );

        if( pr_y == NULL )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:missingInputs",
                "Structure input: Field 'y' required.");

        if( !mxIsDouble(pr_y) )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Vector of double required.");

        if( mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector required.");

        data->n = mxGetM(pr_y);
        data->m = mxGetM(pr_y);
        data->y = mxGetDoubles(pr_y);
    }
    else
    {
        if( !mxIsDouble(prhs) || mxGetN(prhs) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static
void draw_y__theta_x( double *x, Parameter *theta_y, Data *data )
{
    int t,j;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *mu = theta_y->mu_tm;
    double *sigma = theta_y->sigma_tm;
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
        data->y[t] = exp(0.5*x[t]) * ( mu[k] + sigma[k]*rng_gaussian() );
    }
}

static
void log_f_y__theta_x(double *x, Parameter *theta_y, Data *data, double *log_f)
{
    int t,j;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *mu = theta_y->mu_tm;
    double *sigma = theta_y->sigma_tm;
    
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
    
    for( j=0; j<m; j++ )
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
void compute_derivatives_t( Theta *theta, Data *data, int t, double x, double *psi_t )
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *mu = theta->y->mu_tm;
    double *sigma = theta->y->sigma_tm;
    
    derivative(data->y[t], x, m, p, mu, sigma, psi_t);
}

static
void compute_derivatives( Theta *theta, State *state, Data *data )
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *mu = theta->y->mu_tm;
    double *sigma = theta->y->sigma_tm;
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
    mix_gaussian_SV.n_partials_t = n_partials_t;
    mix_gaussian_SV.n_partials_tp1 = n_partials_tp1;
    
    mix_gaussian_SV.usage_string = usage_string;
    
    mix_gaussian_SV.initializeData = initializeData;
    mix_gaussian_SV.initializeTheta = initializeTheta;
    mix_gaussian_SV.initializeParameter = initializeParameter;
    
    mix_gaussian_SV.draw_y__theta_x = draw_y__theta_x;
    mix_gaussian_SV.log_f_y__theta_x = log_f_y__theta_x;
    
    mix_gaussian_SV.compute_derivatives_t = compute_derivatives_t;
    mix_gaussian_SV.compute_derivatives = compute_derivatives;
}