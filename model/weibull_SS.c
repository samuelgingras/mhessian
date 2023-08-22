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
"Name: weibull_SS \n"
"Description: Weibull multiplicative error model\n"
"Extra parameters: \n"
"\t eta \t \n";

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_eta = mxGetField( prhs, 0, "eta" );
    
    // Check for missing parameter
    if( pr_eta == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'eta' required.");

    // Check parameter
    if( !mxIsScalar(pr_eta) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Scalar parameter required.");

    if( mxGetScalar(pr_eta) < 0.0 )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Positive parameter required.");

    // Read model parameter
    theta_y->eta = mxGetScalar(pr_eta);
    theta_y->lambda = exp( lgamma(1 + 1/theta_y->eta) );
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

        if( !mxIsDouble(pr_y) || mxGetN(pr_y) != 1 )
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");
        
        data->n = mxGetM(pr_y);
        data->m = mxGetM(pr_y);
        data->y = mxGetDoubles(pr_y);
    }
    else
    {
        if( !mxIsDouble(prhs) || mxGetN(prhs) != 1 )
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
    int n = data->n;
    double scale = 1/theta_y->lambda;
    double shape = 1/theta_y->eta;

    for( int t=0; t<n; t++ ) {
        double u = rng_exp(1);
        data->y[t] = exp(x[t]) * scale * pow(u,shape);
    }
}

static
void log_f_y__theta_x(double *x, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double eta = theta_y->eta;
    double lambda = theta_y->lambda;

    *log_f = n * (log(eta) + eta * log(lambda));

    for(int t=0; t<n; t++)
    {
        double y_x_t = data->y[t] * exp(-x[t]) * lambda;
        *log_f += (eta - 1) * log(data->y[t]) - pow(y_x_t, eta) - eta * x[t] ;
    }
}

static inline
void derivative(double y_t, double eta, double lambda, double x_t, double *psi_t)
{   
    double eta2 = eta  * eta;
    double eta3 = eta2 * eta;
    double eta4 = eta3 * eta;
    double eta5 = eta4 * eta;

    double y = y_t * exp(-x_t) * lambda;
    double z = pow(y,eta);

    psi_t[1] =  eta  * z - eta;
    psi_t[2] = -eta2 * z;
    psi_t[3] =  eta3 * z;
    psi_t[4] = -eta4 * z;
    psi_t[5] =  eta5 * z;
}

static
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    derivative( data->y[t], theta->y->eta, theta->y->lambda, x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int n = state->n;
    double *x = state->alC;

    for(int t=0; t<n; t++)
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], theta->y->eta, theta->y->lambda, x[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model weibull_SS = {"weibull_SS", initializeModel, 0 };

static
void initializeModel()
{
    weibull_SS.n_theta = n_theta;
    weibull_SS.n_partials_t = n_partials_t;
    weibull_SS.n_partials_tp1 = n_partials_tp1;
    
    weibull_SS.usage_string = usage_string;
    
    weibull_SS.initializeData = initializeData;
    weibull_SS.initializeParameter = initializeParameter;
    
    weibull_SS.draw_y__theta_x = draw_y__theta_x;
    weibull_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    weibull_SS.compute_derivatives_t = compute_derivatives_t;
    weibull_SS.compute_derivatives = compute_derivatives;
}