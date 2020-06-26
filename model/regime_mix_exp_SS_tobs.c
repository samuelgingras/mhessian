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
"Name: regime_mix_exp_SS_tobs\n"
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
    data->y = mxGetDoubles(pr_y);                               // Observation vector
    data->s = mxGetDoubles(pr_s);                               // Regime indicator
    data->k = mxGetDoubles(pr_k);                               // State indicator  
    data->m = (int) mxGetDoubles(pr_k)[data->n - 1];            // Nb of state
    data->pos = (int *) mxMalloc((data->n + 1) * sizeof(int));  // Position indicator
    
    // Compute position indicator
    int i,j;
    data->pos[data->m] = mxGetM(pr_k);
    for(i=0, j=0; i < data->m; i++) {
        while(mxGetPr(pr_k)[j] <= (double) i) {
            j++;
        }
        data->pos[i] = j;
    }
}

static inline 
double f_y__theta_alpha_t(double alpha_t, double y_t, int m, double *p, double *lambda)
{
    double f = 0.0;
    double yp = y_t + 1.0;
    double ym = ((int)y_t == 0) ? 0.0 : (y_t - 1.0);
    
    for(int j=0; j<m; j++)
    {
        double lambda_j = lambda[j] * exp(-alpha_t);
        f += p[j] * (exp(-lambda_j * ym) - exp(-lambda_j * yp));
    }
    
    return 0.5 * f;
}

static
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    double p_t;
    *log_f = 0.0;
    
    for( t=0; t < n; t++ )
    {
        double alpha_t = alpha[(int)data->k[t] - 1];
        if( (int)data->s[t] ){
            p_t = f_y__theta_alpha_t(alpha_t, data->y[t], m, p, lambda);
            *log_f += log(p_t);
        }
    }
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int t;
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    double f;
    double u;
    
    for(t=0; t<n; t++)
    {        
        data->y[t] = 0.0;
        f = f_y__theta_alpha_t(alpha[t], data->y[t], m, p, lambda);
        u =  rng_rand();
        
        while( f < u )
        {
            data->y[t] += 1.0;
            f += f_y__theta_alpha_t(alpha[t], data->y[t], m, p, lambda);
        }
    }
}

static inline void derivative(int m, double *p, double *lambda, double alpha_t, int t, int *pos, double *y, double *s, double *psi_t)
{
    int i,j,d;
    
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    for(i = pos[t]; i < pos[t+1]; i++) {
        
        int s_i = (int) s[i];
        double y_i = y[i];
        
        
        if( s_i ) {
            
            double ym_i = ((int)y_i == 0) ? 0.0 : (y_i - 1.0);
            double yp_i = y_i + 1.0;
            
            double p_it[6] = { 0 };
            double h_it[6] = { 0 };
            double f_it[6] = { 0 };
            
            for(j=0; j<m; j++) {
                
                double lambda_jt = lambda[j] * exp(-alpha_t);
                
                // Step 1a: Direct computation f(x) = exp(exp(-x) * k1)
                double f[6];
                double lnf = -lambda_jt * ym_i;
                f[0] = exp(lnf);
                f[1] = -( f[0] ) * lnf;
                f[2] =  ( f[0] - f[1] ) * lnf;
                f[3] = -( f[0] - 2*f[1] + f[2] ) * lnf;
                f[4] =  ( f[0] - 3*f[1] + 3*f[2] - f[3] ) * lnf;
                f[5] = -( f[0] - 4*f[1] + 6*f[2] - 4*f[3] + f[4] ) * lnf;
                
                
                // Step 1b: Direct computation g(x) = exp(exp(-x) * k2)
                double g[6];
                double lng = -lambda_jt * yp_i;
                g[0] = exp(lng);
                g[1] = -( g[0] ) * lng;
                g[2] =  ( g[0] - g[1] ) * lng;
                g[3] = -( g[0] - 2*g[1] + g[2] ) * lng;
                g[4] =  ( g[0] - 3*g[1] + 3*g[2] - g[3] ) * lng;
                g[5] = -( g[0] - 4*g[1] + 6*g[2] - 4*g[3] + g[4] ) * lng;
                
                // Step 1c: Direct computation f_i(x) = a[j] * (f(x) - g(x))
                for(d=0; d<6; d++)
                    f_it[d] += p[j] * (f[d] - g[d]);
            }
            
            // Step 2: Faa di Bruno p_i(x) = log(f_i(x))
            double z = f_it[0];
            double z_2 = z * z;
            double z_3 = z_2 * z;
            double z_4 = z_3 * z;
            double z_5 = z_4 * z;
            
            h_it[0] = log(z);
            h_it[1] =  1.0 / z;
            h_it[2] = -1.0 / z_2;
            h_it[3] =  2.0 / z_3;
            h_it[4] = -6.0 / z_4;
            h_it[5] = 24.0 / z_5;
            
            compute_Faa_di_Bruno(5, h_it, f_it, p_it);
            
            // Step 3: Direct computation psi = sum_i p_i(x)
            for(d=1; d<6; d++)
                psi_t[d] += p_it[d];
        }
    }
}


static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;
    
    derivative(m, p, lambda, alpha, t, data->pos, data->y, data->s, psi_t);
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
        derivative(m, p, lambda, alpha[t], t, data->pos, data->y, data->s, psi_t);
}


static 
void initializeModel(void);

Observation_model regime_mix_exp_SS_tobs = { initializeModel, 0 };

static 
void initializeModel()
{
    regime_mix_exp_SS_tobs.n_theta = n_theta;
    regime_mix_exp_SS_tobs.n_partials_t = n_partials_t;
    regime_mix_exp_SS_tobs.n_partials_tp1 = n_partials_tp1;
    
    regime_mix_exp_SS_tobs.usage_string = usage_string;
    
    regime_mix_exp_SS_tobs.initializeData = initializeData;
    regime_mix_exp_SS_tobs.initializeTheta = initializeTheta;
    regime_mix_exp_SS_tobs.initializeParameter = initializeParameter;
    
    regime_mix_exp_SS_tobs.draw_y__theta_alpha = draw_y__theta_alpha;
    regime_mix_exp_SS_tobs.log_f_y__theta_alpha = log_f_y__theta_alpha;
    regime_mix_exp_SS_tobs.compute_derivatives_t = compute_derivatives_t;
    regime_mix_exp_SS_tobs.compute_derivatives = compute_derivatives;
}