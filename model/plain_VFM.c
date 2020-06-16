#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"

static int is_mixture = 0;
static int is_factor = 1;
static int n_theta = 1;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string = 
"Name: plain_VFM\n"
"Description: Simple variance factor model with Gaussian innovation\n"
"Extra parameters: for each series of observation j=1,...,J\n"
"\tb_j\t Scale parameter of the jth series\n";

static void initialize_theta_y(const int nrhs, const mxArray *prhs[], int p, Theta *theta)
{
    theta->y = (Parameter *) mxMalloc(sizeof(Parameter));
    
    ErrMsgTxt(nrhs >= p + n_scalar,
    "Invalid input argument: model parameter expected");
    
    ErrMsgTxt(!mxIsChar(prhs[p]),
    "Invalid input argument: model parameter expected");
    
    theta->y->n = n_scalar;
    theta->y->m = length(prhs[p]);
    theta->y->scalar = (double *) mxMalloc(theta->y->m * sizeof(double));
    memcpy(theta->y->scalar, mxGetPr(prhs[p]), theta->y->m * sizeof(double));
}

static void read_data(const mxArray *prhs, Theta *theta, State *state, Data *data)
{
    data->n = length(prhs);
    data->m = mxGetM(pr_y);
    data->y = mxGetPr(prhs);
    
    ErrMsgTxt(theta->y->m * state->n == data->n,
    "Invalid input argument: incompatible observation vector");
}

static void draw_y__theta_alpha(Theta *theta, State *state, Data *data)
{
    int t,n = state->n;
    int j,m = theta->y->m;
    double *alpha = state->alpha;
    double *b = theta->y->scalar;
    
    for(j = 0; j < m; j++)
    {
        for(t = 0; t < n; t++)
            data->y[t + j*n] = exp(b[j] * alpha[t] / 2) * rng_gaussian();
    }
    
}

static void log_f_y__theta_alpha(Theta *theta, State *state, Data *data, double *log_f)
{
    int t,n = state->n;
    int j,m = theta->y->m;
    double *alpha = state->alpha;
    double *b = theta->y->scalar;
    double result = 0.0;
    
    for(j = 0; j < m; j++)
    {
        for(t = 0; t < n; t++)
        {
            double y_jt_2 = data->y[t + j*n] * data->y[t + j*n];
            double b_alpha_jt = b[j] * alpha[t];
            result -= b_alpha_jt + y_jt_2 * exp(-b_alpha_jt);
        }
    }
    result -= m*n * log(2 * M_PI);
    
    *log_f = 0.5 * result;
}

static inline void derivative(double y_t, double alpha_t, int m, double *b, double *psi_t)
{
    for(int j=0; j < m; j++)
    {
        int z_jt = 0.5 * y_t * y_t * exp(-b[j] * alpha_t) * b[j];
        psi_t[1] -= 0.5 * b[j] - z_jt;
        psi_t[2] -= z_jt * b[j];
        psi_t[3] -= psi_t[2] * b[j];
        psi_t[4] -= psi_t[3] * b[j];
        psi_t[5] -= psi_t[4] * b[j];
    }
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    int m = theta->y->m;
    double *b = theta->y->scalar;
    derivative(data->y[t*m], alpha, m, b, psi_t);
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t;
    int n = state->n;
    int m = theta->y->m;
    double *alpha = state->alC;
    double *b = theta->y->scalar;
    double *psi_t;
    
    for(t = 0, psi_t = state->psi; t < n; t++ ,psi_t += state->psi_stride)
        derivative(data->y[t*m], alpha[t], m, b, psi_t);
}

static void initialize(void);

Observation_model plain_VFM = { initialize, 0 };

static void initialize(void)
{

    plain_VFM.n_partials_t = n_partials_t;
    plain_VFM.n_partials_tp1 = n_partials_tp1;
    
    plain_VFM.usage_string = usage_string;
    
    plain_VFM.initialize_theta_y = initialize_theta_y;
    plain_VFM.initialize_data_eval = initialize_data_eval;
    plain_VFM.initialize_data_draw = initialize_data_draw;
    
    plain_VFM.draw_y__theta_alpha = draw_y__theta_alpha;
    plain_VFM.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    plain_VFM.compute_derivatives_t = compute_derivatives_t;
    plain_VFM.compute_derivatives = compute_derivatives;
}