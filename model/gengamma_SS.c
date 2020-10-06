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
"Name: gengamma_SS \n"
"Description: Generalized Gamma multiplicative error model\n"
"Extra parameters: \n"
"\t eta \t \n"
"\t kappa \t "
"\t lambda \t \n";


static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_eta = mxGetField( prhs, 0, "eta" );
    mxArray *pr_kappa = mxGetField( prhs, 0, "kappa" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
    
    // Check for missing parameter
    if( pr_eta == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'eta' required.");

    if( pr_kappa == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'kappa' required.");

    if( pr_lambda == NULL) 
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'lambda' required.");

    // Check parameter
    if( !mxIsScalar(pr_eta) || !mxIsScalar(pr_kappa) || !mxIsScalar(pr_lambda) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameters: Scalar parameters required.");

    if( mxGetScalar(pr_eta) < 0.0 || mxGetScalar(pr_kappa) < 0.0 || mxGetScalar(pr_lambda) < 0.0 )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameters: Positive parameters required.");

    // Read model parameter
    theta_y->eta = mxGetScalar(pr_eta);
    theta_y->kappa = mxGetScalar(pr_kappa);
    theta_y->lambda = mxGetScalar(pr_lambda);
    // theta_y->lambda = exp(lgamma(theta_y->kappa + 1/theta_y->eta) - lgamma(theta_y->kappa));
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
            "Nested structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
            "Nested structure input: Field 'y' required.");

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

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int n = data->n;
    double eta = theta_y->eta;
    double kappa = theta_y->kappa;
    double scale = 1/theta_y->lambda;
    double shape = 1/theta_y->eta;
    
    for(int t=0; t<n; t++)
    {
        double u = rng_gamma(kappa,1);
        data->y[t] = exp(alpha[t]) * scale * pow(u,shape);
    }
}

static void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double eta = theta_y->eta;
    double kappa = theta_y->kappa;
    double lambda = theta_y->lambda;
    double eta_kappa = eta * kappa;

    *log_f = n * (log(eta) - lgamma(kappa) + eta_kappa * log(lambda));

    for(int t=0; t<n; t++)
    {
        double y_alpha_t = data->y[t] * exp(-alpha[t]) * lambda;
        *log_f += (eta_kappa - 1) * log(data->y[t]) - pow(y_alpha_t, eta) - eta_kappa * alpha[t];
    }
}

static inline 
void derivative(double y_t, double eta, double kappa, double lambda, double alpha_t, double *psi_t)
{   
    double eta2 = eta  * eta;
    double eta3 = eta2 * eta;
    double eta4 = eta3 * eta;
    double eta5 = eta4 * eta;

    double y = y_t * exp(-alpha_t) * lambda;
    double z = pow(y,eta);
    
    psi_t[1] =  eta  * z - eta * kappa;
    psi_t[2] = -eta2 * z;
    psi_t[3] =  eta3 * z;
    psi_t[4] = -eta4 * z;
    psi_t[5] =  eta5 * z;
}

static 
void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;

    derivative( data->y[t], eta, kappa, lambda, alpha, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int n = state->n;
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;
    double *alpha = state->alC;

    for(int t=0; t<n; t++)
    {
        double *psi_t = state->psi + t * state->psi_stride;
        derivative( data->y[t], eta, kappa, lambda, alpha[t], psi_t );
    }
}

static
void initializeModel(void);

Observation_model gengamma_SS = { initializeModel, 0 };

static
void initializeModel()
{
    gengamma_SS.n_theta = n_theta;
    gengamma_SS.n_partials_t = n_partials_t;
    gengamma_SS.n_partials_tp1 = n_partials_tp1;
    
    gengamma_SS.usage_string = usage_string;
    
    gengamma_SS.initializeData = initializeData;
    gengamma_SS.initializeTheta = initializeTheta;
    gengamma_SS.initializeParameter = initializeParameter;
    
    gengamma_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    gengamma_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    gengamma_SS.compute_derivatives_t = compute_derivatives_t;
    gengamma_SS.compute_derivatives = compute_derivatives;
}