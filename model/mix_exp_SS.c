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
"Name: exp_SCD\n"
"Description: Linear combination of Exponential densitites for duration modeling\n"
"Extra parameters: for j=1,...,J\n"
"\tw_j\t Component weight of the jth exponential distribution\n"
"\tlambda_j\t Shape parameter of the jth exponential distribution";


static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_w = mxGetField(prhs,0,"w");
    mxArray *pr_l = mxGetField(prhs,0,"lambda");
    
    ErrMsgTxt( pr_w != NULL || pr_l != NULL,
        "Invalid input argument: two model paramters expected");
    
    mxCheckVector(pr_w);
    mxCheckVector(pr_l);
    
    ErrMsgTxt( mxGetM(pr_w) == mxGetM(pr_l),
        "Invalid input argument: incompatible vector length");
    
    theta_y->n = n_theta;
    theta_y->m = mxGetM(pr_w);
    theta_y->scalar = (double *) mxMalloc(theta_y->n * theta_y->m * sizeof( double ));
    memcpy(theta_y->scalar, mxGetPr(pr_w), theta_y->m * sizeof( double ));
    memcpy(theta_y->scalar + theta_y->m, mxGetPr(pr_l), theta_y->m * sizeof( double ));
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
    int t, n = data->n;
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
    double w[m];
    double cumul[m];
    double cte = 0.0;
    
    // Compute weight and cumulative weight of proposal
    for(int j=0; j<m; j++)
    {
        if(a[j] > 0.0)
            w[j] = a[j];
        else
            w[j] = 0.0;
        
        cumul[j] = cte + w[j];
        cte += w[j];
    }
    
    // Accept/Reject algorithm
    for(t=0; t<n; t++) {
        for(;;) 
        {
            // Draw a mixture component
            int k = 0;
            double u = rng_rand();
            while(cumul[k] < u * cte)
                k++;
            
            // Draw y_t_star from proposal
            double mu = lambda[k] * exp(-alpha[t]);
            double y_t_star = rng_exp( 1/mu );
            
            // Evaluate log likelihood
            double log_f = log_f_y__theta_alpha_t(m, a, lambda, y_t_star, alpha[t]);
            double log_g = log_f_y__theta_alpha_t(m, w, lambda, y_t_star, alpha[t]);
            
            // Accept/Reject
            if( rng_rand() < exp(log_f - log_g - log(cte)) ) 
            {
                data->y[t] = y_t_star;
                break;
            }
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

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *lambda = theta_y->scalar + m;
    
    *log_f = 0.0;
    
    for(int t=0; t<n; t++)
        *log_f += log_f_y__theta_alpha_t(m, a, lambda, data->y[t], alpha[t]);
}

static inline
void derivative(double y_t, double alpha_t, int m, double *a, double *lambda, double *psi_t)
{
    double h_jt[6];
    double g_jt[6];
    
    double f_t[6];
    double p_t[6] = { 0 };
    
    for(int j=0; j<m; j++)
    {
        // Step 1: Direct computation
        h_jt[3] = h_jt[5] = lambda[j] * exp(-alpha_t) * y_t;
        h_jt[2] = h_jt[4] = -h_jt[3];
        h_jt[1] = -1 + h_jt[3];
        h_jt[0] = -alpha_t - h_jt[3];
        
        // Step 2: Faa di Bruno with f(x) = exp(x)
        f_t[0] = f_t[1] = f_t[2] = exp(h_jt[0]);
        f_t[3] = f_t[4] = f_t[5] = exp(h_jt[0]);
        compute_Faa_di_Bruno(5, f_t, h_jt, g_jt);
        
        // Step 3: Direct computation
        for(int d=0; d<6; d++)
                p_t[d] +=  a[j] * lambda[j] * g_jt[d];
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

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *a = theta->y->scalar;
    double *lambda = theta->y->scalar + m;
    
    derivative(data->y[t], alpha, m, a, lambda, psi_t);
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int m = theta->y->m;
    double *a = theta->y->scalar;
    double *lambda = theta->y->scalar + m;
    
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(data->y[t], alpha[t], m, a, lambda, psi_t);
}

static void initialize(void);

Observation_model exp_SCD = { initialize, 0 };

static void initialize()
{
    exp_SCD.n_partials_t = n_partials_t;
    exp_SCD.n_partials_tp1 = n_partials_tp1;
    
    exp_SCD.usage_string = usage_string;
    
    exp_SCD.initializeParameter = initializeParameter;
    exp_SCD.read_data = read_data;
    
    exp_SCD.draw_y__theta_alpha = draw_y__theta_alpha;
    exp_SCD.log_f_y__theta_alpha = log_f_y__theta_alpha;
    exp_SCD.compute_derivatives_t = compute_derivatives_t;
    exp_SCD.compute_derivatives = compute_derivatives;
}