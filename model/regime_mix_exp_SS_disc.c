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
"Name:     regime_mix_exp_SS_discm\n"
"Description: Discretized mixture of exponential duration with regime change\n"
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

static inline double f_y__theta_alpha_t(double alpha_t, double y_t, int m, double *a, double *lambda)
{
    double f = 0.0;
    double yp = y_t + 1.0;
    double ym = ((int)y_t == 0) ? 0.0 : (y_t - 1.0);
    
    for(int j=0; j<m; j++)
    {
        double lambda_j = lambda[j] * exp(-alpha_t);
        f += a[j] * (exp(-lambda_j * ym) - exp(-lambda_j * yp));
    }
    
    return 0.5 * f;
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t;
    int n = data->n;
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
    double p_t;
    *log_f = 0.0;
    
    for( t=0; t < n; t++ )
    {
        double alpha_t = alpha[(int)data->k[t] - 1];
        if( (int)data->z[t] ){
            p_t = f_y__theta_alpha_t(alpha_t, data->y[t], m, a, lambda);
            *log_f += log(p_t);
        }
    }
}

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int t;
    int n = data->n;
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
    double f;
    double u;
    
    for(t=0; t<n; t++)
    {
        // mexPrintf("Drawing obs: %d \t state: %1.3f\n",t, alpha[t]);
        
        data->y[t] = 0.0;
        f = f_y__theta_alpha_t(alpha[t], data->y[t], m, a, lambda);
        u =  rng_rand();
        
        while( f < u )
        {
            data->y[t] += 1.0;
            // mexPrintf("\tIteration %1.0f \t value of f=%1.3f \n",data->y[t], f);
            f += f_y__theta_alpha_t(alpha[t], data->y[t], m, a, lambda);
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

static inline void derivative(int m, double *a, double *lambda, double alpha_t, int t, int *p, double *y, double *z, double *psi_t)
{
    int i,j,d;
    
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    for(i = p[t]; i < p[t+1]; i++) {
        
        int z_i = (int) z[i];
        double y_i = y[i];
        
        
        if( z_i ) {
            
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
                    f_it[d] += a[j] * (f[d] - g[d]);
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

Observation_model regime_mix_exp_SS_disc = { initialize, 0 };

static void initialize()
{
    regime_mix_exp_SS_disc.n_partials_t = n_partials_t;
    regime_mix_exp_SS_disc.n_partials_tp1 = n_partials_tp1;
    
    regime_mix_exp_SS_disc.usage_string = usage_string;
    
    regime_mix_exp_SS_disc.initializeParameter = initializeParameter;
    regime_mix_exp_SS_disc.read_data = read_data;
    
    regime_mix_exp_SS_disc.draw_y__theta_alpha = draw_y__theta_alpha;
    regime_mix_exp_SS_disc.log_f_y__theta_alpha = log_f_y__theta_alpha;
    regime_mix_exp_SS_disc.compute_derivatives_t = compute_derivatives_t;
    regime_mix_exp_SS_disc.compute_derivatives = compute_derivatives;
}