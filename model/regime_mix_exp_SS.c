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
    mxArray *pr_p = mxGetField( prhs, 0, "p" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
    
    // Check for missing parameters
    if( pr_p == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'p' required.");

    if( pr_lambda == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'lambda' required.");

    // Check parameters
    if( !mxIsDouble(pr_p) && mxGetN(pr_p) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_lambda) && mxGetN(pr_lambda) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( mxGetM(pr_p) != mxGetM(pr_lambda) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_p);
    theta_y->p_tm = mxGetDoubles(pr_p);
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
    data->n = mxGetM(pr_y);                                 // Nb of observation
    data->y = mxGetDoubles(pr_y);                           // Observation vector
    data->s = (int *) mxMalloc( data->n * sizeof(int) );    // Regime indicator

    for( int t=0; t<data->n; t++ )
        data->s[t] = (int) mxGetDoubles(pr_s)[t];       
}

static 
double log_f_y__theta_alpha_t(int m, double *p , double *lambda, double y_t, double alpha_t)
{
    double f_t = 0.0;
    for(int j=0; j<m; j++) {
        double g_jt = exp( -alpha_t - lambda[j] * exp(-alpha_t) * y_t );
        f_t += p[j] * lambda[j] * g_jt;
    }
    return log(f_t);
}


static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int k, j, t;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
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
        if(p[j] > 0.0)
            w[j] = p[j];
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
            log_f = log_f_y__theta_alpha_t(m, p, lambda, y_t_star, alpha[t]);
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
    int t;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    *log_f = 0.0;
    for( t=0; t<n; t++ ) 
    {
        int s_t = (int)data->s[t];        
        if( s_t )
            *log_f += log_f_y__theta_alpha_t( m, p, lambda, data->y[t], alpha[t] );
    }
}

static inline
void derivative(double y_t, int s_t, double alpha_t, int m, double *p, double *lambda,
    double *psi_t)
{
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    if( s_t )
    {
        int j,d;
        double h_jt[6];
        double g_jt[6];
        
        double f_t[6];
        double p_t[6] = { 0.0 };
        
        for( j=0; j<m; j++ )
        {
            // Step 1: Direct computation
            h_jt[3] = h_jt[5] = lambda[j] * exp(-alpha_t) * y_t;
            h_jt[2] = h_jt[4] = -h_jt[3];
            h_jt[1] = -1 + h_jt[3];
            h_jt[0] = -alpha_t - h_jt[3];
            
            // Step 2: Faa di Bruno with g(x) = exp(h(x))
            f_t[0] = f_t[1] = f_t[2] = exp(h_jt[0]);
            f_t[3] = f_t[4] = f_t[5] = exp(h_jt[0]);
            compute_Faa_di_Bruno(5, f_t, h_jt, g_jt);
            
            // Step 3: Direct computation
            for( d=0; d<6; d++ )
                    p_t[d] +=  p[j] * lambda[j] * g_jt[d];
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
    }
}


static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;

    derivative( data->y[t], (int)data->s[t], alpha, m, p, lambda, psi_t );
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;

    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative( data->y[t], (int)data->s[t], alpha[t], m, p, lambda, psi_t );
}


static
void initializeModel(void);

Observation_model regime_mix_exp_SS = { initializeModel, 0 };

static
void initializeModel()
{
    regime_mix_exp_SS.n_theta = n_theta;
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