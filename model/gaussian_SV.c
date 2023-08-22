#include <math.h>
#include <string.h>
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "errors.h"

static int n_theta = 0;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: gaussian_SV\n"
"Description: Stochastic volatility model, without leverage, with Gaussian distribution\n"
"Extra parameters: none\n";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // No model parameter to initialize
}

static
void initializeData(const mxArray *prhs, Data *data)
{
    if( mxIsStruct(prhs) )
    {
        mxArray *pr_y = mxGetField( prhs, 0, "y" );

        if( pr_y == NULL )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:missingInputs",
                "Structure input: Field 'y' required.");

        if( !mxIsDouble(pr_y) )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Vector of double required.");

        if( mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector required.");

        data->n = mxGetM(pr_y);
        data->m = mxGetM(pr_y);
        data->y = mxGetDoubles(pr_y);
    }
    else
    {
        if( !mxIsDouble(prhs) && mxGetN(prhs) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static
void draw_y__theta_x(double *x, Parameter *theta_y, Data *data)
{
    int t, n = data->n;
    for (t=0; t<n; t++)
        data->y[t] = exp(x[t]/2) * rng_gaussian();
}

static
void log_f_y__theta_x(double *x, Parameter *theta_y, Data *data, double *log_f)
{
    int t, n = data->n;
    double result = 0.0;
    for(t=0; t<n; t++)
    {
        double y_t_2 = data->y[t] * data->y[t];
        result -= y_t_2 * exp(-x[t]) + x[t];
    }
    result -= n * log(2 * M_PI);
    *log_f = 0.5 * result;
}

static inline
void derivative(double y_t, double x_t, double *psi_t)
{
    psi_t[4] = psi_t[2] = -0.5 * (y_t * y_t) * exp( -x_t );
    psi_t[1] = -psi_t[2] - 0.5;
    psi_t[5] = psi_t[3] = -psi_t[2];
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t) 
{
    derivative(data->y[t], x, psi_t);
}

static
void compute_derivatives( Theta *theta, State *state, Data *data )
{
    int t, n = state->n;
    double *x = state->alC; 
    double *psi_t;
    for(t = 0, psi_t = state->psi; t < n; t++, psi_t += state->psi_stride)
        derivative(data->y[t], x[t], psi_t);
}

static
void initializeModel(void);

Observation_model gaussian_SV = {"gaussian_SV", initializeModel, 0 };

static
void initializeModel()
{
    gaussian_SV.n_theta = n_theta;
    gaussian_SV.n_partials_t = n_partials_t;
    gaussian_SV.n_partials_tp1 = n_partials_tp1;
    
    gaussian_SV.usage_string = usage_string;

    gaussian_SV.initializeData = initializeData;
    gaussian_SV.initializeParameter = initializeParameter;

    gaussian_SV.draw_y__theta_x = draw_y__theta_x;
    gaussian_SV.log_f_y__theta_x = log_f_y__theta_x;
    
    gaussian_SV.compute_derivatives_t = compute_derivatives_t;
    gaussian_SV.compute_derivatives = compute_derivatives;
}