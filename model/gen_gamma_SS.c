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
"Name: gen_gamma_SS \n"
"Description: Generalized Gamma multiplicative error model\n"
"Extra parameters: \n"
"\t eta \t \n"
"\t kappa \t "
"\t lambda \t \n";

static void initializeParameter(const mxArray *prhs, Parameter *theta_y)
{
	mxArray *pr_eta = mxGetField(prhs,0,"eta");
	mxArray *pr_kappa = mxGetField(prhs,0,"kappa");

	ErrMsgTxt( pr_eta != NULL || pr_kappa != NULL,
        "Invalid input argument: two model parameters expected");

	theta_y->n = n_theta;
	theta_y->eta = mxGetScalar(pr_eta);
	theta_y->kappa = mxGetScalar(pr_kappa);
	theta_y->lambda = exp(lgamma(theta_y->kappa + 1/theta_y->eta) - lgamma(theta_y->kappa));
}

static void read_data(const mxArray *prhs, Data *data)
{
	mxArray *pr_y = mxGetField(prhs,0,"y");

	data->n = mxGetM(pr_y);
	data->m = mxGetM(pr_y);
	data->y = mxGetPr(pr_y);
}

static void draw_y__theta_alpha(double *alpha, Parameter *theta_y, Data *data)
{
	int n = data->n;
	double eta = theta_y->eta;
	double kappa = theta_y->kappa;
	double lambda = theta_y->lambda;
	double z = 1/eta;
	double m = 1/lambda;

	for(int t=0; t<n; t++)
	{
		double u = rng_gamma(kappa,1);
		data->y[t] = exp(alpha[t]) * m * pow(u,z);
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

static inline void derivative(double y_t, double eta, double kappa, double lambda, double alpha_t, double *psi_t)
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

static void compute_derivatives_t(Theta *theta, Data *data, int t, double alpha, double *psi_t)
{
	double eta = theta->y->eta;
	double kappa = theta->y->kappa;
	double lambda = theta->y->lambda;

	derivative( data->y[t], eta, kappa, lambda, alpha, psi_t );
}

static void compute_derivatives(Theta *theta, State *state, Data *data)
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

static void initialize(void);

Observation_model gen_gamma_SS = { initialize, 0 };

static void initialize()
{
    gen_gamma_SS.n_partials_t = n_partials_t;
    gen_gamma_SS.n_partials_tp1 = n_partials_tp1;
    
    gen_gamma_SS.usage_string = usage_string;
    
    gen_gamma_SS.initializeParameter = initializeParameter;
    gen_gamma_SS.read_data = read_data;
    
    gen_gamma_SS.draw_y__theta_alpha = draw_y__theta_alpha;
    gen_gamma_SS.log_f_y__theta_alpha = log_f_y__theta_alpha;
    
    gen_gamma_SS.compute_derivatives_t = compute_derivatives_t;
    gen_gamma_SS.compute_derivatives = compute_derivatives;
}