#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"


static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string = 
"Name: mixture_SV\n"
"Description: univarite Stochastic volatility gaussian mixture model, without leverage\n"
"Extra parameters: for each mixture component j=1,..,J\n"
"\tw_j\t Component weight of the jth Gaussian distribution\n"
"\tmu_j\t Mean of the jth Gaussian distribution\n"
"\tsigma_j\t Std of the jth Gaussian distribution\n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_w = mxGetField(prhs,0,"w");
    mxArray *pr_mu = mxGetField(prhs,0,"mu");
    mxArray *pr_sig = mxGetField(prhs,0,"sigma");
    
    ErrMsgTxt( pr_w != NULL || pr_mu != NULL || pr_sig != NULL,
        "Invalid input argument: three model paramters expected");
    
    mxCheckVector(pr_w);
    mxCheckVector(pr_mu);
    mxCheckVector(pr_sig);
    
    ErrMsgTxt( mxGetM(pr_w) == mxGetM(pr_mu) & mxGetM(pr_w) == mxGetM(pr_sig),
        "Invalid input argument: incompatible vector length");
    
    theta_y->n = n_theta;
    theta_y->m = mxGetM(pr_w);
    theta_y->scalar = (double *) mxMalloc(theta_y->n * theta_y->m * sizeof( double ));
    memcpy(theta_y->scalar, mxGetPr(pr_w), theta_y->m * sizeof( double ));
    memcpy(theta_y->scalar + theta_y->m, mxGetPr(pr_mu), theta_y->m * sizeof( double ));
    memcpy(theta_y->scalar + 2*theta_y->m, mxGetPr(pr_sig), theta_y->m * sizeof( double ));
}

static void read_data(const mxArray *prhs, Data *data)
{
    mxArray *pr_y = mxGetField(prhs,0,"y");
    
    ErrMsgTxt( pr_y != NULL,
        "Invalid input argument: data struct: 'y' field missing");
    ErrMsgTxt( mxGetN(pr_y),
        "Invalid input argument: data struct: column vector expected");
    
    data->n = mxGetM(pr_y);
    data->m = mxGetM(pr_y);
    data->y = mxGetPr(pr_y);
}

static void draw_y__theta_alpha( double *alpha, Parameter *theta_y, Data *data )
{
    int t,n = data->n;
    int m = theta_y->m;
    double *w = theta_y->scalar;
    double *mu = theta_y->scalar + m;
    double *sigma = theta_y->scalar + 2*m;
    double cumul[m];
    
    // Compute cumulative weight
    cumul[0] = w[0];
    for(int j=1; j < m; j++)
        cumul[j] = w[j] + cumul[j-1];
    
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

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    int m = theta_y->m;
    double *w = theta_y->scalar;
    double *mu = theta_y->scalar + m;
    double *sigma = theta_y->scalar + 2*m;
    double result = 0.0;
    
    for(t=0; t<n; t++) {
        double p_t = 0.0;
        for(int j = 0; j < m; j++) {
            double mu_jt = mu[j] * exp(alpha[t]/2);
            double sigma_jt = sigma[j] * exp(alpha[t]/2);
            double z_jt = (data->y[t] - mu_jt) / sigma_jt;
            p_t += w[j] * exp(-0.5 *z_jt * z_jt) / sigma_jt; 
        }
        result += log(p_t);
    }
    *log_f = result - n * 0.5 * log(2*M_PI);
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

static inline void derivative(double y_t, double alpha_t, int m, double *w, double *mu, double *sigma, double *psi_t)
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
                p_t[d] = w[j] * h_jt[d] / sigma[j];
            else
                p_t[d] += w[j] * h_jt[d] / sigma[j];
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
    double *w = theta->y->scalar;
    double *mu = theta->y->scalar + m;
    double *sigma = theta->y->scalar + 2*m;
    
    derivative(data->y[t], alpha, m, w, mu, sigma, psi_t);
}

static
void compute_derivatives( Theta *theta, State *state, Data *data )
{
    int m = theta->y->m;
    double *w = theta->y->scalar;
    double *mu = theta->y->scalar + m;
    double *sigma = theta->y->scalar + 2*m;
    double *alpha = state->alC; 
    
    double *psi_t;
    int t, n = state->n;
    
    for(t=0, psi_t = state->psi; t < n; t++, psi_t += state->psi_stride )
        derivative(data->y[t], alpha[t], m, w, mu, sigma, psi_t);
}

static void initialize(void);

Observation_model mixture_SV = { initialize, 0 };

static void initialize()
{
    mixture_SV.n_partials_t = n_partials_t;
    mixture_SV.n_partials_tp1 = n_partials_tp1;
    
    mixture_SV.usage_string = usage_string;
    
    mixture_SV.initializeParameter = initializeParameter;
    mixture_SV.read_data = read_data;
    
    mixture_SV.draw_y__theta_alpha = draw_y__theta_alpha;
    mixture_SV.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    mixture_SV.compute_derivatives_t = compute_derivatives_t;
    mixture_SV.compute_derivatives = compute_derivatives;
}