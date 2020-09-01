
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
"Name: flexible_SCD\n"
"Description: Discretized mixture of exponential duration with regime change\n"
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
    data->m = (int) mxGetDoubles(pr_k)[data->n-1];              // Nb of state
    data->y = mxGetDoubles(pr_y);                               // Observation vector
    data->s = (int *) mxMalloc( data->n * sizeof(int) );        // Regime indicator
    data->k = (int *) mxMalloc( data->n * sizeof(int) );        // State indicator
    data->p = (int *) mxMalloc( data->m * sizeof(int) );        // Position indicator

    int i,j,k_im1 = -1;
    for( i=0, j=0; i<data->n; i++ ) {
        data->s[i] = (int) mxGetDoubles(pr_s)[i];
        data->k[i] = (int) mxGetDoubles(pr_k)[i] - 1;
        if( k_im1 != data->k[i] ) {
            data->p[j] = i;
            k_im1 = data->k[i];
            j++;
        }
    }
}


static double log_f_y__theta_alpha_t(
    int m,
    double *p,
    double *lambda,
    double y_t,
    double alpha_t
    )
{
    double result = 0.0;
    for( int j=0; j<m; j++ ) {
        double g_jt = lambda[j] * exp( -lambda[j] * y_t * exp(-alpha_t) - alpha_t );
        result += p[j] * g_jt;
    }
    return log(result);
}

static void log_f_y__theta_alpha(
    double *alpha,
    Parameter *theta_y,
    Data *data,
    double *log_f
    )
{
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    *log_f = 0.0; 
    for( int i=0; i<data->n; i++ ) {
        if( data->s[i] )
            *log_f += log_f_y__theta_alpha_t( m, p, lambda, data->y[i], alpha[data->k[i]] );
    }
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{

}

static inline void derivative(
    int m,
    double *p,
    double *lambda,
    double alpha_t,
    int t,
    Data *data,
    double *psi_t
    )
{
    
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    // Get index of first and last observation conditional on alpha[t]
    int start = data->p[t];
    int end = ((t+1) < data->m) ? data->p[t+1] : data->n;

    for( int i=start; i<end; i++ ) {
        
        if( data->s[i] ) {
            
            double psi_it[6] = { 0.0 };
            double h_jt[6] = { 0.0 };
            double g_jt[6] = { 0.0 };
            double f_it[6] = { 0.0 };
            double q[6] = { 0.0 };

            for( int j=0; j<m; j++ ) {
                // Step 1: Direct computation
                h_jt[3] = h_jt[5] = lambda[j] * data->y[i] * exp(-alpha_t);
                h_jt[2] = h_jt[4] = -h_jt[3];
                h_jt[1] = h_jt[3] - 1;
                h_jt[0] = h_jt[2] - alpha_t;

                // Step 2: Faa di Bruno with composition q(x) = exp(x)
                q[0] = exp(h_jt[0]);
                q[1] = exp(h_jt[0]);
                q[2] = exp(h_jt[0]);
                q[3] = exp(h_jt[0]);
                q[4] = exp(h_jt[0]);
                q[5] = exp(h_jt[0]);
                compute_Faa_di_Bruno(5, q, h_jt, g_jt);

                // Step 3: Direct computation
                for( int d=0; d<6; d++ )
                    f_it[d] += p[j] * lambda[j] * g_jt[d];
            }
            
            // Step 2: Faa di Bruno with composition q(x) = log(x)
            double z = f_it[0];
            double z_2 = z * z;
            double z_3 = z_2 * z;
            double z_4 = z_3 * z;
            double z_5 = z_4 * z;
            
            q[0] = log(z);
            q[1] =  1.0 / z;
            q[2] = -1.0 / z_2;
            q[3] =  2.0 / z_3;
            q[4] = -6.0 / z_4;
            q[5] = 24.0 / z_5;
            compute_Faa_di_Bruno(5, q, f_it, psi_it);
            
            // Step 3: Direct computation
            for( int d=0; d<6; d++ )
                psi_t[d] += psi_it[d];
        }
    }
}


static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;
    
    derivative( m, p, lambda, alpha, t, data, psi_t );
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;

    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;

    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative( m, p, lambda, alpha[t], t, data, psi_t );
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