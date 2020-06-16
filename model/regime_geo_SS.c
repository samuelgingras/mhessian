#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"


static int n_theta = 0;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name: regime_geo_SS\n"
"Description: Geometric duration with regime change\n"
"Extra parameters: None \n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    theta_y->n = n_theta;
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
    int n = data->n;
    
    *log_f = 0.0;
    for(int t=0; t<n; t++) {
        
        int k_t = (int) data->k[t];
        int z_t = (int) data->z[t];
        
        double alpha_t = alpha[k_t - 1];
        double y_t = data->y[t];
        
        if( z_t )
            *log_f += y_t * alpha_t - (y_t + 1) * log(1 + exp(alpha_t));
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
                            + 15*f[3]*g2_2*g[1] + 10*f[3]*g[3]*g1_2 
                            + 10*f[4]*g[2]*g1_3 + f[5]*g1_5;
                    }
                }
            }
        }
    }
}

static inline
void derivative(double alpha_t, int t, int *p, double *y, double *z, double *psi_t)
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
            
            double h[6];
            double f[6];
            double p[6];
            
            // Step 1: Direct computation h(x) = 1+exp(x)
            h[0] = 1 + exp(alpha_t);
            h[1] = h[2] = h[3] = h[4] = h[5] = h[0] - 1;
            
            
            // Step 2: Faa Di Bruno f(x) = log(h(x))
            double z = h[0];
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
            
            compute_Faa_di_Bruno(5, f, h, p);
            
            // Update derivatives
            psi_t[1] += y_i  - (y_i + 1) * p[1];
            psi_t[2] -= (y_i + 1) * p[2];
            psi_t[3] -= (y_i + 1) * p[3];
            psi_t[4] -= (y_i + 1) * p[4];
            psi_t[5] -= (y_i + 1) * p[5];
        }
    }
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    derivative(alpha, t, data->p, data->y, data->z, psi_t);
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(alpha[t], t, data->p, data->y, data->z, psi_t);
}

static void initialize(void);

Observation_model regime_geo_SS = { initialize, 0 };

static void initialize()
{
    regime_geo_SS.n_partials_t = n_partials_t;
    regime_geo_SS.n_partials_tp1 = n_partials_tp1;
    
    regime_geo_SS.usage_string = usage_string;
    
    regime_geo_SS.initializeParameter = initializeParameter;
    regime_geo_SS.read_data = read_data;
    
    regime_geo_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    regime_geo_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    regime_geo_SS.compute_derivatives_t = compute_derivatives_t;
    regime_geo_SS.compute_derivatives = compute_derivatives;
}