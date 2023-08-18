#include <math.h>
#include <string.h>
#include "errors.h"
#include "mex.h"
#include "RNG.h"
#include "state.h"
#include "faa_di_bruno.h"

static int n_dimension_parameters = 1;
static int n_theta = 2;
static int n_partials_t = 5;
static int n_partials_tp1 = 0;


static char *usage_string =
"Name: mix_exp_SS\n"
"Description: Mixture of exponential densities for duration modeling\n"
"Extra parameters: for j=1,...,J\n"
"\tw_j\t Component weight of the jth exponential distribution\n"
"\tlambda_j\t Shape parameter of the jth exponential distribution";

static int all_positive(int n, double *p) {
    for (int i=0; i<n; i++)
        if (p[i] < 0)
            return 0;
    return 1;
}

static int sum_to_one(int n, double *p) {
    double sum = 0.0;
    for (int i=0; i<n; i++)
        sum += p[i];
    return (fabs(sum-1.0) < 1e-9);
}

static Parameter_specification parameter_specification[] = {
    {"p", 0, -1, sum_to_one},
    {"lambda", 0, -1, all_positive}    
};

static
void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
    // Set pointer to field
    mxArray *pr_p = mxGetField( prhs, 0, "p" );
    mxArray *pr_lambda = mxGetField( prhs, 0, "lambda" );
    
    // Check for missing parameters
    if( pr_p == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'p' required.");

    if( pr_lambda == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input: Field 'lambda' required.");

    // Check parameters
    if( !mxIsDouble(pr_p) || mxGetN(pr_p) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");

    if( !mxIsDouble(pr_lambda) || mxGetN(pr_lambda) != 1)
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Column vector of double required.");
    
    if( mxGetM(pr_p) != mxGetM(pr_lambda) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Model parameter: Incompatible vector length.");

    // Set pointer to theta_y
    theta_y->m = mxGetM(pr_p);
    theta_y->p_tm = mxGetDoubles(pr_p);
    theta_y->lambda_tm = mxGetDoubles(pr_lambda);
}

static
void initializeTheta(const mxArray *prhs, Theta *theta)
{
    // Check structure input
    if( !mxIsStruct(prhs) )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Structure input required.");

    // Check nested structure
    mxArray *pr_theta_x = mxGetField( prhs, 0, "x" );
    mxArray *pr_theta_y = mxGetField( prhs, 0, "y" );

    if( pr_theta_x == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Nested structure input: Field 'x' required.");

    if( pr_theta_y == NULL )
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
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
double log_f_y__theta_alpha_t(int m, double *p , double *lambda, double y_t, double alpha_t)
{
    double p_t = 0.0;
    for(int j=0; j<m; j++) {
        double g_jt = exp( -alpha_t - lambda[j] * exp(-alpha_t) * y_t );
        p_t += p[j] * lambda[j] * g_jt;
    }
    return log(p_t);
}

static
void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
    int t, n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    double w[m];
    double cumul[m];
    double cte = 0.0;
    
    // Compute weight and cumulative weight of proposal
    for(int j=0; j<m; j++)
    {
        if(p[j] > 0.0)
            w[j] = p[j];
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
            double log_f = log_f_y__theta_alpha_t(m, p, lambda, y_t_star, alpha[t]);
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

static
void log_f_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data, double *log_f)
{
    int n = data->n;
    int m = theta_y->m;
    double *p = theta_y->p_tm;
    double *lambda = theta_y->lambda_tm;
    
    *log_f = 0.0;
    for(int t=0; t<n; t++)
        *log_f += log_f_y__theta_alpha_t(m, p, lambda, data->y[t], alpha[t]);
}

static inline
void derivative(double y_t, double alpha_t, int m, double *p, double *lambda, double *psi_t)
{
    double h_jt[6];
    double g_jt[6];
    
    double f_t[6];
    double p_t[6] = { 0.0 };
    
    for(int j=0; j<m; j++)
    {
        // Step 1: Direct computation
        h_jt[3] = h_jt[5] = lambda[j] * exp(-alpha_t) * y_t;
        h_jt[2] = h_jt[4] = -h_jt[3];
        h_jt[1] = -1 + h_jt[3];
        h_jt[0] = -alpha_t - h_jt[3];
        
        // Step 2: Faa di Bruno with g(x) = exp(h(x))
        f_t[0] = f_t[1] = f_t[2] = exp(h_jt[0]);
        f_t[3] = f_t[4] = f_t[5] = exp(h_jt[0]);
        compute_Faa_di_Bruno(5, f_t, h_jt, g_jt);
        
        // Step 3: Direct computation
        for(int d=0; d<6; d++)
                p_t[d] +=  p[j] * lambda[j] * g_jt[d];
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
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;
    
    derivative(data->y[t], alpha, m, p, lambda, psi_t);
}

static
void compute_derivatives(Theta *theta, State *state, Data *data)
{
    int m = theta->y->m;
    double *p = theta->y->p_tm;
    double *lambda = theta->y->lambda_tm;
    
    int t, n = state->n;
    double *alpha = state->alC;
    double *psi_t;
    
    for(t=0, psi_t = state->psi; t<n; t++, psi_t += state->psi_stride )
        derivative(data->y[t], alpha[t], m, p, lambda, psi_t);
}

static
void initializeModel(void);

Observation_model mix_exp_SS = { initializeModel, 0 };

static
void initializeModel()
{
    mix_exp_SS.n_theta = n_theta;
    mix_exp_SS.n_partials_t = n_partials_t;
    mix_exp_SS.n_partials_tp1 = n_partials_tp1;
    
    mix_exp_SS.usage_string = usage_string;
    
    mix_exp_SS.initializeData = initializeData;
    mix_exp_SS.initializeTheta = initializeTheta;
    mix_exp_SS.initializeParameter = initializeParameter;
    
    mix_exp_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    mix_exp_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;

    mix_exp_SS.compute_derivatives_t = compute_derivatives_t;
    mix_exp_SS.compute_derivatives = compute_derivatives;
}