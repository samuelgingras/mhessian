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
"Name: regime_mix_geo_SS\n"
"Description: Geometric duration with regime change\n"
"Extra parameters: None \n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    mxArray *pr_w = mxGetField(prhs,0,"w");
    mxArray *pr_m = mxGetField(prhs,0,"mu");
    
    ErrMsgTxt( pr_w != NULL || pr_m != NULL,
    "Invalid input argument::model::two parameters expected");
    
    mxCheckVector(pr_w);
    mxCheckVector(pr_m);
    
    ErrMsgTxt( mxGetM(pr_w) == mxGetM(pr_m),
    "Invalid input argument::model::incompatible vector length");
    
    theta_y->n = n_theta;
    theta_y->m = mxGetM(pr_w);
    
    int pos[2] = {0,theta_y->m};
    theta_y->scalar = (double *) mxMalloc(theta_y->n * theta_y->m * sizeof(double));
    memcpy(theta_y->scalar + pos[0], mxGetPr(pr_w), theta_y->m * sizeof(double));
    memcpy(theta_y->scalar + pos[1], mxGetPr(pr_m), theta_y->m * sizeof(double));
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


static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{

}


static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int m = theta_y->m;
    double *a = theta_y->scalar;
    double *mu = theta_y->scalar + m;
    
    *log_f = 0.0;
    for(int t=0; t<data->n; t++) {
        
        int k_t = (int) data->k[t];
        int z_t = (int) data->z[t];
        
        double alpha_t = alpha[k_t - 1];
        double y_t = data->y[t];
        
        if( z_t ) {
            double p_t = 0.0;
            for(int j=0; j<m; j++) {
                double mu_tj = mu[j] * exp(alpha_t);
                double lambda_tj = 1 / (1+mu_tj);
                double h_tj = pow((1-lambda_tj), y_t) * lambda_tj;
                p_t += a[j] * h_tj;
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


static double compute_Leibniz(int n, double *f, double *g)
{
    double fg;
    
    switch (n)
    {
        case 0:
        fg = f[0]*g[0];
        break;
        
        case 1:
        fg = f[1]*g[0] + f[0]*g[1];
        break;
        
        case 2:
        fg = f[2]*g[0] + 2*f[1]*g[1] + f[0]*g[2];
        break;
        
        case 3:
        fg = f[3]*g[0] + 3*f[2]*g[1] + 3*f[1]*g[2] + f[0]*g[3];
        break;
        
        case 4:
        fg = f[4]*g[0] + 4*f[3]*g[1] + 6*f[2]*g[2] + 4*f[1]*g[3] + f[0]*g[4];
        break;
        
        case 5:
        fg = f[5]*g[0] + 5*f[4]*g[1] + 10*f[3]*g[2] + 10*f[2]*g[3];
        fg = fg + 5*f[1]*g[4] + f[0]*g[5];
        break;
        
        default:
        fg = 0.0;
        break;
    }
    return fg;
}


static inline
void derivative(int m, double *a, double *mu, double alpha_t, int t, int *p, double *y, double *z, double *psi_t)
{
    psi_t[1] = 0.0;
    psi_t[2] = 0.0;
    psi_t[3] = 0.0;
    psi_t[4] = 0.0;
    psi_t[5] = 0.0;
    
    
    for(int i = p[t]; i < p[t+1]; i++) {
        
        int z_i = (int) z[i];
        int y_i = y[i];
        
        if( z_i ) {
            
            double f[6];                // For composition using Faa Di Bruno
            double h_tij[6];            // Evaluation for state t, obs i, mixture j
            double g_tij[6];            // Evaluation for state t, obs i, mixture j
            double g_tj[6];             // Evaluation for state t, mixture j
            double h_ti[6] = {0};       // Evaluation for state t, obs i
            double psi_ti[6] = {0};     // Derivative for state t, obs i
            
            for(j=0; j<m; j++)
            {
                // Step 1: Direct computation g(x) = lambda * exp(x)
                double mu_tj = mu[j] * exp(alpha_t);
                g_tj[0] = g_tj[1] = g_tj[2] = g_tj[3] = g_tj[4] = g_tj[5] = mu_tj;
                
                
                // Step 2: Faa Di Bruno g_tij(x) = (1+g_tj(x))^(-(y_i+1))
                double z_inv = 1 / (1-mu_tj);
                f[0] = pow(z_inv, y_i+1);
                f[1] = -(y_i+1) * z_inv * f[0];
                f[2] = -(y_i+2) * z_inv * f[1];
                f[3] = -(y_i+3) * z_inv * f[2];
                f[4] = -(y_i+4) * z_inv * f[3];
                f[5] = -(y_i+5) * z_inv * f[4];
                
                compute_Faa_di_Bruno(5, f, g_tj, g_tij);
                
                
                // Step 3: Leibniz rule h(x) = f(x) * g(x)
                h_tij[0] = compute_Leibniz(0, g_tij, g_tj);
                h_tij[1] = compute_Leibniz(1, g_tij, g_tj);
                h_tij[2] = compute_Leibniz(2, g_tij, g_tj);
                h_tij[3] = compute_Leibniz(3, g_tij, g_tj);
                h_tij[4] = compute_Leibniz(4, g_tij, g_tj);
                h_tij[5] = compute_Leibniz(5, g_tij, g_tj);
                
                
                // Step 4: Direct computation for mixture components
                h_ti[0] = h_ti[0] + a[j] * h_tij[0];
                h_ti[1] = h_ti[1] + a[j] * h_tij[1];
                h_ti[2] = h_ti[2] + a[j] * h_tij[2];
                h_ti[3] = h_ti[3] + a[j] * h_tij[3];
                h_ti[4] = h_ti[4] + a[j] * h_tij[4];
                h_ti[5] = h_ti[5] + a[j] * h_tij[5];
            }
            
            
            // Step 5: Faa Di Bruno psi_ti(x) = log(h_ti(x))
            double z_1 = h_ti[0];
            double z_2 = z_1 * z_1;
            double z_3 = z_2 * z_1;
            double z_4 = z_3 * z_1;
            double z_5 = z_4 * z_1;
            
            f[0] = log(z);
            f[1] =  1.0 / z;
            f[2] = -1.0 / z_2;
            f[3] =  2.0 / z_3;
            f[4] = -6.0 / z_4;
            f[5] = 24.0 / z_5;
            
            compute_Faa_di_Bruno(5, f, h_ti, psi_ti);
            
            
            // Step 6: Direct computation psi_t = sum_i psi_ti
            psi_t[1] += psi_ti[1];
            psi_t[2] += psi_ti[2];
            psi_t[3] += psi_ti[3];
            psi_t[4] += psi_ti[4];
            psi_t[5] += psi_ti[5];
        }
    }
}


static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y-m;
    double *a = theta->y->scalar;
    double *mu = theta->y->scalar + m;
    
    derivative(m, a, mu, alpha, t, data->p, data->y, data->z, psi_t);
}


static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    int m = theta->y-m;
    double *a = theta->y->scalar;
    double *mu = theta->y->scalar + m;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(m, a, mu, alpha[t], t, data->p, data->y, data->z, psi_t);
}


static void initialize(void);

Observation_model regime_mix_geo_SS = { initialize, 0 };

static void initialize()
{
    regime_mix_geo_SS.n_partials_t = n_partials_t;
    regime_mix_geo_SS.n_partials_tp1 = n_partials_tp1;
    
    regime_mix_geo_SS.usage_string = usage_string;
    
    regime_mix_geo_SS.initializeParameter = initializeParameter;
    regime_mix_geo_SS.read_data = read_data;
    
    regime_mix_geo_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    regime_mix_geo_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    regime_mix_geo_SS.compute_derivatives_t = compute_derivatives_t;
    regime_mix_geo_SS.compute_derivatives = compute_derivatives;
}