#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"

static int n_theta = 3;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;

static char *usage_string = 
"Name: burr_SS \n"
"Description: Burr multiplicative error model\n"
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
    initializeThetax( pr_theta_x, theta->x );
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

static void draw_y__theta_x(double *x, Parameter *theta_y, Data *data)
{
    int n = data->n;
    double shape_eta = 1/theta_y->eta;
    double shape_kappa = 1/theta_y->kappa;
    double scale_lambda = 1/theta_y->lambda;
    
    for( int t=0; t<n; t++ ) {
        double u = rng_rand();
        double v = pow( 1/(1-u), shape_kappa );
        data->y[t] = exp(x[t]) * scale_lambda * pow( v-1, shape_eta );
    }
}

static void log_f_y__theta_x(double *x, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    double eta = theta_y->eta;
    double kappa = theta_y->kappa;
    double lambda = theta_y->lambda;

    *log_f = n * ( eta * log(lambda) + log(eta) + log(kappa) );

    for( int t=0; t<n; t++ ) {
        double z = lambda * data->y[t] * exp(-x[t]);
        double z_eta = pow( z, eta );
        *log_f += (eta-1) * log(data->y[t]) - eta * x[t] - (kappa+1) * log1p(z_eta);
    }
}

static inline 
void derivative(double eta, double kappa, double lambda, double y_t, double x_t, double *psi_t)
{   
    double g[6];
    double h[6];
    double q[6];

    // Step 1: Direct computation h(x) = (lambda * y * exp(-x))^eta;
    h[0] = pow( lambda * y_t * exp(-x_t), eta );
    h[1] = -h[0] * eta;
    h[2] = -h[1] * eta;
    h[3] = -h[2] * eta;
    h[4] = -h[3] * eta;
    h[5] = -h[4] * eta;

    // Step 2a: Faa Di Bruno g(x) = q(h(x)) with q(x) = log(1+x);
    double z = 1+h[0];
    double z_inv = 1/z;
    q[0] = log(z);
    q[1] = z_inv;
    q[2] = q[1] * z_inv * (-1.0);
    q[3] = q[2] * z_inv * (-2.0);
    q[4] = q[3] * z_inv * (-3.0);
    q[5] = q[4] * z_inv * (-4.0);
    compute_Faa_di_Bruno( 5, q, h, g );

    // Step 2b: Direct computation of psi(x) = -eta*x - (kappa+1)*g(x)
    psi_t[1] = -(kappa+1) * g[1] - eta;
    psi_t[2] = -(kappa+1) * g[2];
    psi_t[3] = -(kappa+1) * g[3];
    psi_t[4] = -(kappa+1) * g[4];
    psi_t[5] = -(kappa+1) * g[5];

}

static 
void compute_derivatives_t(Theta *theta, Data *data, int t, double x, double *psi_t)
{
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;

    derivative( eta, kappa, lambda, data->y[t], x, psi_t );
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    
    double eta = theta->y->eta;
    double kappa = theta->y->kappa;
    double lambda = theta->y->lambda;
    
    int t, n = state->n;
    double *x = state->alC;
    double *psi_t;

    for( t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative( eta, kappa, lambda, data->y[t], x[t], psi_t );
}

static
void initializeModel(void);

Observation_model burr_SS = { initializeModel, 0 };

static
void initializeModel()
{
    burr_SS.n_theta = n_theta;
    burr_SS.n_partials_t = n_partials_t;
    burr_SS.n_partials_tp1 = n_partials_tp1;
    
    burr_SS.usage_string = usage_string;
    
    burr_SS.initializeData = initializeData;
    burr_SS.initializeTheta = initializeTheta;
    burr_SS.initializeParameter = initializeParameter;
    
    burr_SS.draw_y__theta_x = draw_y__theta_x;
    burr_SS.log_f_y__theta_x = log_f_y__theta_x;
    
    burr_SS.compute_derivatives_t = compute_derivatives_t;
    burr_SS.compute_derivatives = compute_derivatives;
}