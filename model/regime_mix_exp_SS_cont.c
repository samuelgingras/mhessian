/*
This file implement the regime switching stochastic conditional duration model. The state variables is updated in each second and all transaction recorded between two consecutives second are associated to the same state variables. The duration distribution is specified for continuous observation.

    Data input:
        y       complete vector of observed duration
        z       regime indicator
        k       index of each duration and state i.e. k=j -> x_j | y_k
*/


#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"


static int n_theta = 2;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name:     regime_mix_exp_SS_cont\n"
"Description: Exponential duration with regime change\n"
"Extra parameters: for j=1,...,J\n"
"\tw_j\t Component weight of the jth exponential distribution\n"
"\tlambda_j\t Shape parameter of the jth exponential distribution";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_w = mxGetField(prhs,0,"w");
    mxArray *pr_l = mxGetField(prhs,0,"lambda");
    
    ErrMsgTxt( pr_w != NULL || pr_l != NULL,
    "Invalid input argument: model : four model paramters expected");
    
    mxCheckVector(pr_w);
    mxCheckVector(pr_l);
    
    ErrMsgTxt( mxGetM(pr_w) == mxGetM(pr_l),
    "Invalid input argument: model argument : incompatible vector length");
    
    theta_y->n = n_theta;
    theta_y->m = mxGetM(pr_w);
    
    int pos[2] = {0,theta_y->m};
    theta_y->scalar = (double *) mxMalloc(theta_y->n * theta_y->m * sizeof(double));
    memcpy(theta_y->scalar + pos[0], mxGetPr(pr_w), theta_y->m * sizeof(double));
    memcpy(theta_y->scalar + pos[1], mxGetPr(pr_l), theta_y->m * sizeof(double));
}


static void read_data(const mxArray *prhs, Data *data)
{
    mxArray *pr_y = mxGetField(prhs,0,"y");
    mxArray *pr_z = mxGetField(prhs,0,"z");
    mxArray *pr_k = mxGetField(prhs,0,"k");
    
    ErrMsgTxt( pr_y != NULL,
    "Invalid input argument: data struct: 'y' field missing");
    ErrMsgTxt( mxGetN(pr_y),
    "Invalid input argument: data struct: column vector expected");
    
    ErrMsgTxt( pr_z != NULL,
    "Invalid input argument: data struct: 'z' field missing");
    ErrMsgTxt( mxGetN(pr_y),
    "Invalid input argument: data struct: column vector expected");
    
    ErrMsgTxt( pr_k != NULL,
    "Invalid input argument: data struct: 'k' field missing");
    ErrMsgTxt( mxGetN(pr_k),
    "Invalid input argument: data struct: column vector expected");
    
    ErrMsgTxt( mxGetM(pr_y) == mxGetM(pr_z),
    "Invalid input argument: data struct: incompatible vector length");
    
    ErrMsgTxt( mxGetM(pr_y) == mxGetM(pr_k),
    "Invalid input argument: data struct: incompatible vector length");
    
    
    // Fill in data structure
    data->n = mxGetM(pr_y);                                     // Nb of observation
    data->m = (int) mxGetPr(pr_k)[data->n - 1];                 // Nb of state
    data->y = mxGetPr(pr_y);                                    // Duration vector
    data->z = mxGetPr(pr_z);                                    // Regime indicator
    data->k = mxGetPr(pr_k);                                    // State indicator
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

static double log_f_y__theta_alpha_t(int m, double *a , double *lambda, double y_t, double alpha_t)
{
    double p_t = 0.0;
    for(int j=0; j<m; j++) {
        double g_jt = exp( -alpha_t - lambda[j] * exp(-alpha_t) * y_t );
        p_t += a[j] * lambda[j] * g_jt;
    }
    return log(p_t);
}


static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int k, j, t;
    int n = data->n;
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
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
            w[j] = a[j];
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
            log_f = log_f_y__theta_alpha_t(m, a, lambda, y_t_star, alpha[t]);
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
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
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
                p_t += a[j] * lambda[j] * g_jt;
            }
            
            *log_f += log(p_t);
        }
    }
}


static void compute_Faa_di_Bruno(int n, double *f, double *g, double *fg)
{
    fg[0] = f[0];
    if( n >= 1) {
        fg[1] = f[1]*g[1];
        if( n >= 2 ) {
            double g1_2 = g[1]*g[1];
            fg[2] = f[1]*g[2] + f[2]*g1_2;
            if( n >= 3 ) {
                double g1_3 = g1_2*g[1];
                fg[3] = f[1]*g[3] + 3*f[2]*g[1]*g[2] + f[3]*g1_3;
                if( n >= 4 ) {
                    double g2_2 = g[2]*g[2];
                    double g1_4 = g1_3*g[1];
                    fg[4] = f[1]*g[4] + 4*f[2]*g[1]*g[3]
                        + 3*f[2]*g2_2 + 6*f[3]*g1_2*g[2] + f[4]*g1_4;
                    if( n >= 5 ) {
                        double g1_5 = g1_4*g[1];
                        fg[5] = f[1]*g[5] + 5*f[2]*g[1]*g[4] + 10*f[2]*g[2]*g[3]
                            + 15*f[3]*g2_2*g[1] + 10*f[3]*g[3]*g1_2 + 10*f[4]*g[2]*g1_3
                                + f[5]*g1_5;
                    }
                }
            }
        }
    }
}


static inline
void derivative(int m, double *a, double *lambda, double alpha_t, int t, int *p, double *y, double *z, double *psi_t)
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
                        g_ti[d] +=  a[j] * lambda[j] * g_tij[d];
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
    double *a = theta->y->scalar;
    double *lambda = theta->y->scalar + m;
    
    derivative(m, a, lambda, alpha, t, data->p, data->y, data->z, psi_t);
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    int m = theta->y->m;
    double *a = theta->y->scalar;
    double *lambda = theta->y->scalar + m;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(m, a, lambda, alpha[t], t, data->p, data->y, data->z, psi_t);
}


static void initialize(void);

Observation_model regime_mix_exp_SS_cont = { initialize, 0 };

static void initialize()
{
        regime_mix_exp_SS_cont.n_partials_t = n_partials_t;
        regime_mix_exp_SS_cont.n_partials_tp1 = n_partials_tp1;
    
        regime_mix_exp_SS_cont.usage_string = usage_string;
    
        regime_mix_exp_SS_cont.initializeParameter = initializeParameter;
        regime_mix_exp_SS_cont.read_data = read_data;
    
        regime_mix_exp_SS_cont.draw_y__theta_alpha = draw_y__theta_alpha;
        regime_mix_exp_SS_cont.log_f_y__theta_alpha = log_f_y__theta_alpha;
        regime_mix_exp_SS_cont.compute_derivatives_t = compute_derivatives_t;
        regime_mix_exp_SS_cont.compute_derivatives = compute_derivatives;
}