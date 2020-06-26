#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"

static int n_theta = 2;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string =
"Name: regime_mix_exp_SS\n"
"Description: Exponential duration with regime change\n"
"Extra parameters: for j=1,...,J\n"
"\tw_j\t Component weight of the jth exponential distribution\n"
"\tlambda_j\t Shape parameter of the jth exponential distribution";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_pi = mxGetField( prhs, 0, "pi" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
    
    // Check for missing parameters
    if( pr_pi == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'pi' required.");

    if( pr_lambda == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'lambda' required.");

    // Check parameters
    if( !mxIsDouble(pr_pi) && mxGetN(pr_pi) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_lambda) && mxGetN(pr_lambda) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( mxGetM(pr_pi) != mxGetM(pr_lambda) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_pi);
    theta_y->pi_tm = mxGetDoubles(pr_pi);
    theta_y->lambda_tm = mxGetDoubles(pr_lambda);
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
    mxArray *pr_k = mxGetField( prhs, 0, "k" );
    
    // Check for missing inputs
    if( pr_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 'y' with observation required.");

    if( pr_s == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 's' with regime index required.");

    if( pr_k == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Field 'k' with state index required.");
    
    // Check inputs
    if( !mxIsDouble(pr_y) && mxGetN(pr_y) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( !mxIsDouble(pr_s) && mxGetN(pr_s) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( !mxIsDouble(pr_k) && mxGetN(pr_k) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Column vectors of type double required.");

    if( mxGetM(pr_y) != mxGetM(pr_s) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Incompatible vectors length.");

    if( mxGetM(pr_y) != mxGetM(pr_k) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Data: Incompatible vectors length.");
    
    
    // Fill in data structure
    data->n = mxGetM(pr_y);                                     // Nb of observation
    data->y = mxGetDoubles(pr_y);                               // Observation vector
    data->s = mxGetDoubles(pr_s);                               // Regime indicator
    data->k = mxGetDoubles(pr_k);                               // State indicator  
    data->m = (int) mxGetDoubles(pr_k)[data->n - 1];            // Nb of state
    data->p = (int *) mxMalloc((data->n + 1) * sizeof(int));    // Position indicator
    
    // Compute position indicator
    int i,j;
    data->p[data->m] = mxGetM(pr_k);
    for(i=0, j=0; i < data->m; i++) {
        while(mxGetPr(pr_k)[j] <= (double) i) {
            j++;
        }
        data->p[i] = j;
    }
}

static 
double log_f_y__theta_alpha_t(int m, double *pi , double *lambda, double y_t, double alpha_t)
{
    double p_t = 0.0;
    for(int j=0; j<m; j++) {
        double g_jt = exp( -alpha_t - lambda[j] * exp(-alpha_t) * y_t );
        p_t += pi[j] * lambda[j] * g_jt;
    }
    return log(p_t);
}


static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int k, j, t;
    int n = data->n;
    int m = theta_y->m;
    double *pi = theta_y->pi_tm;
    double *lambda = theta_y->lambda_tm;
    
    double w[m];
    double cumul[m];
    double cte = 0.0;
    
    double u;
    double log_f;
    double log_g;
    double lambda_k;
    double y_t_star;
    
    // Compute weight and cumulative weight of proposal
    for( j=0; j<m; j++ )
    {
        if(a[j] > 0.0)
            w[j] = pi[j];
        else
            w[j] = 0.0;
        
        cumul[j] = cte + w[j];
        cte += w[j];
    }
    
    // Accept/Reject algorithm
    for( t=0; t<n; t++ ) {
        for(;;) 
        {
            // Draw a mixture component
            k = 0;
            u = rng_rand();
            while( cumul[k] < u * cte )
                k++;
            
            // Draw y_t_star from proposal
            lambda_k = lambda[k] * exp(-alpha[t]);
            y_t_star = rng_exp( 1/lambda_k );
            
            // Evaluate log likelihood
            log_f = log_f_y__theta_alpha_t(m, pi, lambda, y_t_star, alpha[t]);
            log_g = log_f_y__theta_alpha_t(m, w, lambda, y_t_star, alpha[t]);
            
            // Accept/Reject
            if( rng_rand() < exp(log_f - log_g - log(cte)) )
            {
                data->y[t] = y_t_star;
                break;
            }
        }
    }
}


static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int m = theta_y->m;
    double *pi = theta_y->pi_tm;
    double *lambda = theta_y->lambda_tm;
    
    *log_f = 0.0;
    for(int t=0; t < data->n; t++) {
        
        int k_t = (int) data->k[t];
        int z_t = (int) data->z[t];
        
        double alpha_t = alpha[k_t - 1];
        double y_t = data->y[t];
        
        if( z_t ) {
            double p_t = 0.0;
            for(int j=0; j < m; j++) {
                double g_jt = exp(-alpha_t - lambda[j] * exp(-alpha_t) * y_t);
                p_t += pi[j] * lambda[j] * g_jt;
            }
            
            *log_f += log(p_t);
        }
    }
}

static inline
void derivative(int m, double *pi, double *lambda, double alpha_t, int t, int *p, double *y,
    double *z, double *psi_t)
{
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    int i,j,d;
    
    for(i = p[t]; i < p[t+1]; i++) {
        
        int z_i = (int) z[i];       // Regime indicator obs i
        double y_i = y[i];          // Duration obs i
        
        if( z_i ) {
            double f[6];                // For composition using Faa di Bruno
            double h_tij[6];            // Evaluation for state t, obs i, mixture j
            double g_tij[6];            // Evaluation for state t, obs i, mixture j
            double g_ti[6] = { 0 };     // Evaluation for state t, obs i
            double psi_ti[6] = { 0 };   // Derivative for state t, obs i
            
            for(j=0; j<m; j++) 
            {
                // Step 1: Direct computation
                h_tij[3] = h_tij[5] = lambda[j] * exp(-alpha_t) * y_i;
                h_tij[2] = h_tij[4] = -h_tij[3];
                h_tij[1] = -1 + h_tij[3];
                h_tij[0] = -alpha_t - h_tij[3];
                
                // Step 2: Faa di Bruno with f(x) = exp(x)
                f[0] = f[1] = f[2] = exp(h_tij[0]);
                f[3] = f[4] = f[5] = exp(h_tij[0]);
                compute_Faa_di_Bruno(5, f, h_tij, g_tij);
                
                // Step 3: Direct computation
                for(d=0; d<6; d++)
                        g_ti[d] +=  pi[j] * lambda[j] * g_tij[d];
            }
            
            // Step 4: Faa di Bruno with f(x) = log(x)
            double z = g_ti[0];
            double z_2 = z * z;
            double z_3 = z_2 * z;
            double z_4 = z_3 * z;
            double z_5 = z_4 * z;
            
            f[0] = log(z);
            f[1] =  1.0 / z;
            f[2] = -1.0 / z_2;
            f[3] =  2.0 / z_3;
            f[4] = -6.0 / z_4;
            f[5] = 24.0 / z_5;
            
            compute_Faa_di_Bruno(5, f, g_ti, psi_ti);
            
            // Step 5: Direct computation psi_t = sum_i psi_ti(x)
            for(d=1; d<6; d++) {
                psi_t[d] += psi_ti[d];
            }
        }
    }
}


static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *pi = theta_y->pi_tm;
    double *lambda = theta_y->lambda_tm;

    derivative(m, pi, lambda, alpha, t, data->p, data->y, data->z, psi_t);
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    int m = theta->y->m;
    double *pi = theta_y->pi_tm;
    double *lambda = theta_y->lambda_tm;

    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(m, pi, lambda, alpha[t], t, data->p, data->y, data->z, psi_t);
}


static
void initializeModel(void);

Observation_model regime_mix_exp_SS = { initializeModel, 0 };

static
void initializeModel()
{
    regime_mix_exp_SS.n_theta = n_theta
    regime_mix_exp_SS.n_partials_t = n_partials_t;
    regime_mix_exp_SS.n_partials_tp1 = n_partials_tp1;

    regime_mix_exp_SS.usage_string = usage_string;

    regime_mix_exp_SS.initializeData = initializeData;
    regime_mix_exp_SS.initializeTheta = initializeTheta;
    regime_mix_exp_SS.initializeParameter = initializeParameter;

    regime_mix_exp_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    regime_mix_exp_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    regime_mix_exp_SS.compute_derivatives_t = compute_derivatives_t;
    regime_mix_exp_SS.compute_derivatives = compute_derivatives;
}