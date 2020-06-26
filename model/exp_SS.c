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
"Name: exp_SS\n"
"Description: Exponential duration model\n"
"Extra parameters: none\n";

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{
    // Check if structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input required.");

    // Check if nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );

    if( pr_theta_x != NULL )
        initializeThetaAlpha( pr_theta_x, theta->alpha );
    else
        initializeThetaAlpha( prhs, theta->alpha );
}

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

        if( !mxIsDouble(pr_y) && mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

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
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int t,n = data->n;
    for (t=0; t<n; t++)
        data->y[t] = rng_exp(exp(alpha[t]));
}

static
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int t,n = data->n;
    double result = 0.0;
    for (t=0; t<n; t++)
        result += -alpha[t] - data->y[t] * exp(-alpha[t]);
    
    *log_f = result;
}

static inline
void derivative(double y_t, double alpha_t, double *psi_t)
{
	psi_t[3] = psi_t[5] = y_t * exp(-alpha_t);
	psi_t[2] = psi_t[4] = -psi_t[3];
	psi_t[1] = psi_t[3] - 1.0;
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
	derivative( data->y[t], alpha, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int t, n = state->n;
    double *alpha = state->alC;
    
    for( t=0; t<n; t++ )
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], alpha[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model exp_SS = { initializeModel, 0 };

static
void initializeModel()
{
    exp_SS.n_theta = n_theta;
    exp_SS.n_partials_t = n_partials_t;
    exp_SS.n_partials_tp1 = n_partials_tp1;
    
    exp_SS.usage_string = usage_string;
    
    exp_SS.initializeData = initializeData;    
    exp_SS.initializeTheta = initializeTheta;
    exp_SS.initializeParameter = initializeParameter;
    
    exp_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    exp_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    exp_SS.compute_derivatives_t = compute_derivatives_t;
    exp_SS.compute_derivatives = compute_derivatives;
}