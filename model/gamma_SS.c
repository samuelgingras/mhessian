#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"

static int n_theta = 1;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: gamma_SS \n"
"Description: Gamma multiplicative error model\n"
"Extra parameters: \n"
"\t kappa \t \n";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_kappa = mxGetField( prhs, 0, "kappa" );
    
    // Check for missing parameter
    if( pr_kappa == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'kappa' required.");

    // Check parameter
    if( !mxIsScalar(pr_kappa) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Scalar parameter required.");

    if( mxGetScalar(pr_kappa) < 0.0 )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Positive parameter required.");

    // Read model parameter
    theta_y->kappa = mxGetScalar(pr_kappa);
    theta_y->lambda = theta_y->kappa;
}

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{

    // Check structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input required.");

    // Check nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );
    mxArray *pr_theta_y = mxGetField( prhs, 0, "y" );

    if( pr_theta_x == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Structure input: Field 'y' required.");

    // Read state and model parameters
    initializeThetaAlpha( pr_theta_x, theta->alpha );
    initializeParameter( pr_theta_y, theta->y );

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
            mexErrMsgIdAndTxt( "mhessian:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(prhs);
        data->m = mxGetM(prhs);
        data->y = mxGetDoubles(prhs);
    }
}

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int n = data->n;
    double kappa = theta_y->kappa;
    double scale = 1/theta_y->lambda;
    
    for( int t=0; t<n; t++ )
        data->y[t] = exp(alpha[t]) * scale * rng_gamma(kappa,1);
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double kappa = theta_y->kappa;

    *log_f = n * ( kappa * log(kappa) - lgamma(kappa) );

    for(int t=0; t<n; t++)
    {
        double y_alpha_t = data->y[t] * exp(-alpha[t]) * kappa;
        *log_f += (kappa - 1) * log(data->y[t]) - kappa * alpha[t] - y_alpha_t;
    }
}

static inline void derivative(double y_t, double kappa, double alpha_t, double *psi_t)
{   
    psi_t[3] = psi_t[5] = y_t * exp(-alpha_t) * kappa;
    psi_t[2] = psi_t[4] = -psi_t[3];
    psi_t[1] = psi_t[3] - kappa;
}

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    derivative( data->y[t], theta->y->kappa, alpha, psi_t );
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int n = state->n;
    double *alpha = state->alC;

    for(int t=0; t<n; t++)
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], theta->y->kappa, alpha[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model gamma_SS = { initialize, 0 };

static
void initializeModel()
{
    gamma_SS.n_theta = n_theta;
    gamma_SS.n_partials_t = n_partials_t;
    gamma_SS.n_partials_tp1 = n_partials_tp1;
    
    gamma_SS.usage_string = usage_string;
    
    gamma_SS.initializeData = initializeData;
    gamma_SS.initializeTheta = initializeTheta;
    gamma_SS.initializeParameter = initializeParameter;
    
    gamma_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    gamma_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    gamma_SS.compute_derivatives_t = compute_derivatives_t;
    gamma_SS.compute_derivatives = compute_derivatives;
}