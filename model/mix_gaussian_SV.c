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
    mxArray *pr_pi = mxGetField( prhs, 0, "pi" );
    mxArray *pr_mu = mxGetField( prhs, 0, "mu" );
    mxArray *pr_sigma = mxGetField( prhs, 0, "sigma" );
    
    // Check for missing parameters
    if( pr_pi == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'pi' required.");

    if( pr_mu == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'mu' required.");

    if( pr_sigma == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'sigma' required.");

    // Check parameters
    if( !mxIsDouble(pr_pi) && mxGetN(pr_pi) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_mu) && mxGetN(pr_mu) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( !mxIsDouble(pr_sigma) && mxGetN(pr_sigma) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( mxGetM(pr_pi) != mxGetM(pr_mu) || mxGetM(pr_pi) != mxGetM(pr_sigma) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_pi);
    theta_y->pi_tm = mxGetDoubles(pr_pi);
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
    initializeThetaAlpha( pr_theta_x, theta->alpha );
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
        if( !mxIsDouble(prhs) && mxGetN(prhs) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static
void draw_y__theta_alpha( double *alpha, Parameter *theta_y, Data *data )
{
    int t,n = data->n;
    int m = theta_y->m;
    double *pi = theta_y->pi_tm;
    double *mu = theta_y->mu_tm;
    double *sigma = theta_y->sigma_tm;
    double cumul[m];
    
    // Compute cumulative weight
    cumul[0] = w[0];
    for(int j=1; j < m; j++)
        cumul[j] = pi[j] + cumul[j-1];
    
    for(int t=0; t<n; t++) {
        // Draw component
        int k = 0;
        double u = rng_rand();
        
        while( cumul[k] < u )
            k++;
        
        // Draw data
        data->y[t] = exp(alpha[t]/2) * (mu[k] + sigma[k]*rng_gaussian());
    }
}

static
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    int m = theta_y->m;
    double *pi = theta_y->pi_tm;
    double *mu = theta_y->mu_tm;
    double *sigma = theta_y->sigma_tm;
    double result = 0.0;
    
    for(t=0; t<n; t++) {
        double p_t = 0.0;
        for(int j = 0; j < m; j++) {
            double mu_jt = mu[j] * exp(alpha[t]/2);
            double sigma_jt = sigma[j] * exp(alpha[t]/2);
            double z_jt = (data->y[t] - mu_jt) / sigma_jt;
            p_t += pi[j] * exp(-0.5 *z_jt * z_jt) / sigma_jt; 
        }
        result += log(p_t);
    }
    *log_f = result - n * 0.5 * log(2*M_PI);
}

static inline
void derivative(double y_t, double alpha_t, int m, double *pi, double *mu, double *sigma,
    double *psi_t)
{
    double g_jt[6];
    double h_jt[6];
    
    double f_t[6];
    double p_t[6];
    
    for(int j = 0; j < m; j++)
    {
        // Step 1 : Direct computation 	
        double A = -0.5 / (sigma[j] * sigma[j]);
        double B = y_t * y_t * exp(-alpha_t);
        double C = y_t * mu[j] * exp(-alpha_t/2);
        
        g_jt[0] = A * ( B - 2*C + mu[j]*mu[j]);
        g_jt[1] = A * (-B + C);
        g_jt[2] = A * ( B - 0.5*C);
        g_jt[3] = A * (-B + 0.25*C);
        g_jt[4] = A * ( B - 0.125*C);
        g_jt[5] = A * (-B + 0.0625*C);
        
        
        // Step 2 : Faa di Bruno with f(x) = exp(x)
        f_t[0] = f_t[1] = f_t[2] = exp(g_jt[0]);
        f_t[3] = f_t[4] = f_t[5] = exp(g_jt[0]);
        compute_Faa_di_Bruno(5, f_t, g_jt, h_jt);
        
        
        // Step 3 : Direct computation
        for(int d=0; d<6; d++)
        {
            if(j == 0)
                p_t[d] = pi[j] * h_jt[d] / sigma[j];
            else
                p_t[d] += pi[j] * h_jt[d] / sigma[j];
        }
    }
    
    // Step 4: Faa di Bruno with f(x) = log(x)
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
    
    // Adjust derivatives with the value of the first term
    psi_t[0] -= 0.5 * (log(2*M_PI) + alpha_t);
    psi_t[1] -= 0.5;
}

static void compute_derivatives_t( Theta *theta, Data *data, int t, double alpha, double *psi_t )
{
    int m = theta->y->m;
    double *pi = theta->y->pi_tm;
    double *mu = theta->y->mu_tm;
    double *sigma = theta->y->sigma_tm;
    
    derivative(data->y[t], alpha, m, pi, mu, sigma, psi_t);
}

static
void compute_derivatives( Theta *theta, State *state, Data *data )
{
    int m = theta->y->m;
    double *pi = theta->y->pi_tm;
    double *mu = theta->y->mu_tm;
    double *sigma = theta->y->sigma_tm;
    double *alpha = state->alC; 
    
    double *psi_t;
    int t, n = state->n;
    
    for(t=0, psi_t = state->psi; t < n; t++, psi_t += state->psi_stride )
        derivative(data->y[t], alpha[t], m, pi, mu, sigma, psi_t);
}

static
void initializeModel(void);

Observation_model mix_gaussian_SV = { initializeModel, 0 };

static void initializeModel()
{
    mix_gaussian_SV.n_theta = n_theta;
    mix_gaussian_SV.n_partials_t = n_partials_t;
    mix_gaussian_SV.n_partials_tp1 = n_partials_tp1;
    
    mix_gaussian_SV.usage_string = usage_string;
    
    mix_gaussian_SV.initializeData = initializeData;
    mix_gaussian_SV.initializeTheta = initializeTheta;
    mix_gaussian_SV.initializeParameter = initializeParameter;
    
    mix_gaussian_SV.draw_y__theta_alpha = draw_y__theta_alpha;
    mix_gaussian_SV.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    mix_gaussian_SV.compute_derivatives_t = compute_derivatives_t;
    mix_gaussian_SV.compute_derivatives = compute_derivatives;
}