#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "mex.h"
#include "RNG.h"
#include "skew.h"
#include "state.h"
#include "symmetric_Hermite.h"


#define TRUE    1
#define FALSE   0

// --------------------------------------------------------------------------------------------- //
// For diagnostic                                                                                //
// --------------------------------------------------------------------------------------------- //

/*
 The following macro definitions are for reporting errors in HESSIAN method code.
 The result of calling the verify macro is to report a Matlab warning, and pass control back to
 Matlab as quickly as possible. We report a Matlab warning rather than an error, in order for
 Matlab to return a structure containing information useful to track down the problem.
 
 Example of using verify macro:
 
    verify(isfinite(skew.z), "Draw of alpha_t is infinite or not a number", "t=%d, n=%d", t, n);
 
 This should give a result such as:
    Problem detected at line 278 of C source code file alpha_univariate.c, in function draw_HESSIAN
    Iteration information: t=2390, n=2391
    Problem: Draw of alpha_t is infinite or not a number
 */

#define MatlabWarningFormat \
"Problem detected at line %d of C source code file %s, in function %s\n" \
"Iteration information: %s\n" \
"Problem: %s"
static char iter_string[1000];
// static int fatal_error_detected = FALSE;   // Add to state


#define verify(condition, warn_string, iter_fmt_string, ...) \
if (!(condition)) { \
sprintf(iter_string, iter_fmt_string, __VA_ARGS__); \
mexWarnMsgIdAndTxt("HESSIAN:generalError", MatlabWarningFormat, __LINE__, __FILE__, __func__, iter_string, warn_string); \
}

// fatal_error_detected = TRUE; \

// The following code computes diagnostic values to help find efficiency problems. These values are not
// used for any required computation.
void compute_diagnostics(int t, State *state, Skew_parameters *skew)
{
    // sigma is 1 over the square root of minus the 2nd derivative of the approximation of the log
    // conditional density of alpha_t given alpha_{t+1}, y_1, ..., y_t.
    double sigma_m2 = -skew->h2;      // sigma^-2
    double sigma_2 = 1.0 / sigma_m2;  // sigma^2
    double sigma = sqrt(sigma_2);
    double sigma_3 = sigma_2 * sigma; // sigma^3
    double coeff = 0.0;
    double *psi_t = state->psi + t*state->psi_stride;

    // Contribution of contemporaneous observation to conditional precision
    // Should be between zero and one.
    // This is not necessarily a good measure of distortion as log f(y_t|alpha_t) could be close to quadratic
    state->psi2ratio[t] = -psi_t[2] * sigma_2;
    
    // Change in log density for a positive one-sigma change in x attributable to ...
    // (High absolute values indicate high levels of distortion, signs are meaningful.)
    coeff = sigma_3 / 6.0;
    state->h3norm[t] = skew->h3 * coeff;           // ... 3rd order term
    state->psi3norm[t] = psi_t[3] * coeff;   // ... contemporaneous part of 3rd order term
    coeff = sigma_2 * sigma_2 / 24.0;
    state->h4norm[t] = skew->h4 * coeff;           // ... 4th order term
    state->psi4norm[t] = psi_t[4] * coeff;   // ... contemporaneous part of 4th order term
    coeff = sigma_3 * sigma_2 / 120.0;
    state->h5norm[t] = skew->h5 * coeff;           // ... 5th order term
    state->psi5norm[t] = psi_t[5] * coeff;   // ... contemporaneous part of 5th order term
    
    // Ratio of conditional prior variance to 1st order conditional posterior variance.
    // Should be greater than 1, high values indicate high levels of distortion
    state->s2priornorm[t] = state->Sigma_prior[t] * sigma_m2;
}

// --------------------------------------------------------------------------------------------- //
// Computation                                                                                   //
// --------------------------------------------------------------------------------------------- //

void alpha_prior_draw( State_parameter *theta_alpha, double *alpha )
{
    int t, n = theta_alpha->n;
    if( theta_alpha->is_basic )
    {
        double phi = theta_alpha->phi;
        double stddev = sqrt( 1.0/theta_alpha->omega );
        
        alpha[0] = (stddev / sqrt(1-phi*phi)) * rng_gaussian();
        
        for( t=1; t<n; t++ )
            alpha[t] = phi * alpha[t-1] + stddev * rng_gaussian();
        for( t=0; t<n; t++ )
            alpha[t] += theta_alpha->alpha_mean;
    }
    else
    {
        double *d_tm = theta_alpha->d_tm;
        double *omega_tm = theta_alpha->omega_tm;
        double *phi_tm = theta_alpha->phi_tm;
        double stddev = 1.0 / sqrt( omega_tm[0] );
        
        alpha[0] = d_tm[0] + stddev * rng_gaussian();
        
        for( t=1; t<n; t++ ) {
            stddev = 1.0 / sqrt( omega_tm[t] );
            alpha[t] = d_tm[t] + phi_tm[t] * alpha[t-1] + stddev * rng_gaussian();
        }
    }
}

void alpha_prior_eval( State_parameter *theta_alpha, double *alpha, double *log_p )
{
    int t,n = theta_alpha->n;
    double result;
    
    if( theta_alpha->is_basic )
    {
        double alpha_mean = theta_alpha->alpha_mean;
        double phi = theta_alpha->phi;
        double omega = theta_alpha->omega;
        double diff_t = alpha[0] - alpha_mean;
        
        result = omega * (1-phi*phi) * diff_t * diff_t;
        
        for( t=1; t<n; t++ )
        {
            double diff_tm1 = diff_t;
            diff_t = alpha[t] - alpha_mean;
            result += omega * (diff_t - phi * diff_tm1) * (diff_t - phi * diff_tm1);
        }
        
        *log_p = 0.5 * (n*log(omega) + log(1-phi*phi) - n*log(2*M_PI) - result);
    }
    else
    {
        double log_det;
        double *d_tm = theta_alpha->d_tm;
        double *omega_tm = theta_alpha->omega_tm;
        double *phi_tm = theta_alpha->phi_tm;
        double diff_t = alpha[0] - d_tm[0];
        
        result = omega_tm[0] * diff_t * diff_t;
        log_det = log( omega_tm[0] );
        
        for( t=1; t<n; t++ ) {
            diff_t = alpha[t] - d_tm[t] - phi_tm[t] * alpha[t-1];
            result += omega_tm[t] * diff_t * diff_t;
            log_det += log( omega_tm[t] );
        }
        
        *log_p = 0.5 * (log_det - n*log(2*M_PI) - result);
    }
}

static 
void make_Hb_cb( State_parameter *theta_alpha, State *state )
{
    int t,n = state->n;
	double *Hb_0 = state->Hb_0, *Hb_1 = state->Hb_1, *cb = state->cb;
	
    if( theta_alpha->is_basic ) {
        double phi = theta_alpha->phi;
        double omega = theta_alpha->omega;
        double alpha_mean = theta_alpha->alpha_mean;
        
        double Hb_tt = omega * (1+phi*phi);
        double Hb_ttp1 = -omega * phi;
        double cb_t = omega * (1-phi)*(1-phi) * alpha_mean;
        
        Hb_0[0] = Hb_0[n-1] = omega;
        Hb_1[0] = Hb_ttp1;
        cb[0] = cb[n-1] = omega * (1-phi) * alpha_mean;
        
        for( t=1; t<n-1; t++ ) {
            Hb_0[t] = Hb_tt;
            Hb_1[t] = Hb_ttp1;
            cb[t] = cb_t;
        }
    }
    else {
        double *phi_tm = theta_alpha->phi_tm;
        double *omega_tm = theta_alpha->omega_tm;
        double *d_tm = theta_alpha->d_tm;
        
        cb[0] = omega_tm[0] * d_tm[0];
        
        for( t=0; t<n-1; t++ ) {
            Hb_0[t] = omega_tm[t] + omega_tm[t+1] * phi_tm[t+1]* phi_tm[t+1];
            Hb_1[t] = -omega_tm[t+1] * phi_tm[t+1];
            cb[t] -= phi_tm[t+1] * omega_tm[t+1] * d_tm[t+1];
            cb[t+1] = omega_tm[t+1] * d_tm[t+1];
        }
        
        Hb_0[n-1] = omega_tm[n-1];
    }
}

static
void make_Hbb_cbb( Observation_model *model, Theta *theta, State *state, Data *data )
{
    int t, n = state->n, stride = state->psi_stride;
	double *Hb_0 = state->Hb_0, *Hb_1 = state->Hb_1, *cb = state->cb;
	double *Hbb_0 = state->Hbb_0, *cbb = state->cbb;
	double *Hbb_1 = state->Hbb_1, *Hbb_1_2 = state->Hbb_1_2;
	double *psi_t = state->psi, *alC = state->alC;
	
    model->compute_derivatives( theta, state, data );
    
	if( model->n_partials_tp1 > 0 ) {
		int Q = model->n_partials_tp1 + 1;
		double psi_t_01_prev = 0.0;
		double psi_t_02_prev = 0.0;
		double H_1_t_prev    = 0.0;
		double almudd_prev   = 0.0;
		
		for( t=0; t<n-1; t++ )
		{
			psi_t = state->psi + t*stride;
			double H_0_t = -(psi_t[2*Q] + psi_t_02_prev);
			double H_1_t = -psi_t[Q+1];
			double c_t   =  H_1_t_prev*almudd_prev + H_0_t*alC[t] + H_1_t*alC[t+1]
			+ psi_t[Q] + psi_t_01_prev;
			
			Hbb_0[t] = Hb_0[t] + H_0_t;
			cbb[t] =  cb[t] + c_t;
			Hbb_1[t] = Hb_1[t] + H_1_t;
			Hbb_1_2[t] = Hbb_1[t] * Hbb_1[t];
			
			psi_t_01_prev = psi_t[1];
			psi_t_02_prev = psi_t[2];
			H_1_t_prev = H_1_t;
			almudd_prev = alC[t];
		}
		psi_t += stride;
		double H_0_t = -(psi_t[2*Q] + psi_t_02_prev);
		double c_t   = H_1_t_prev*almudd_prev + H_0_t*alC[t]+ psi_t[Q] + psi_t_01_prev;
		Hbb_0[t]     = Hb_0[t] + H_0_t;
		cbb[t]       =  cb[t] + c_t;
	}
	else {
		for( t=0; t<n; t++, psi_t += stride ) {
			Hbb_0[t] = Hb_0[t] - psi_t[2];
			cbb[t] = cb[t] + psi_t[1] - psi_t[2] * alC[t];
		}
		for( t=0; t<n-1; t++ ) {
			Hbb_1[t] = Hb_1[t];
			Hbb_1_2[t] = Hbb_1[t] * Hbb_1[t];
		}
	}
}

static 
void compute_prior_Sigma_ad( State *state )
{
    int t, n = state->n;
    double *Hb_0 = state->Hb_0, *cb = state->cb;
    double *Hb_1 = state->Hb_1;
    double *Sigma = state->Sigma_prior, *m = state->m_prior, *ad = state->ad_prior;
    
    Sigma[0] = 1.0 / Hb_0[0];
    m[0] = Sigma[0] * cb[0];
    ad[0] = -Sigma[0] * Hb_1[0];
    
    for( t=1; t<n; t++ )
    {
        Sigma[t] = 1.0  / (Hb_0[t] - Hb_1[t-1]*Hb_1[t-1] * Sigma[t-1]);
        m[t] = Sigma[t] * (cb[t] - Hb_1[t-1] * m[t-1]);
        ad[t] = -Sigma[t] * Hb_1[t];
    }
}

static 
void compute_Sigma_m( State *state )
{
    int t, n = state->n;
	double *Hbb_0 = state->Hbb_0, *cbb = state->cbb;
	double *Hbb_1 = state->Hbb_1, *Hbb_1_2 = state->Hbb_1_2;
	double *Sigma = state->Sigma, *m = state->m;
	
    Sigma[0] = 1.0 / Hbb_0[0];
    m[0] = Sigma[0] * cbb[0];
    
    for( t=1; t<n; t++ )
    {
        Sigma[t] = 1.0  / (Hbb_0[t] - Hbb_1_2[t-1] * Sigma[t-1]);
        m[t] = Sigma[t] * (cbb[t] - Hbb_1[t-1] * m[t-1]);
	}
}

static 
void compute_Sigma_m_plus( State *state )
{
    int t, n = state->n;
	double *Hbb_0 = state->Hbb_0, *cbb = state->cbb;
	double *Hbb_1 = state->Hbb_1, *Hbb_1_2 = state->Hbb_1_2;
	double *Sigma = state->Sigma, *m = state->m;
	double *s = state->s, *ad = state->ad;
	double x = Hbb_0[0];
    double y = -log(x);
    
    s[0] = y;
    Sigma[0] = exp( s[0] );
    ad[0] = -Sigma[0] * Hbb_1[0];
    m[0] = Sigma[0] * cbb[0];
    
    for( t=1; t<n; t++ )
    {
        s[t] = -log( Hbb_0[t] - Hbb_1_2[t-1] * Sigma[t-1] );
        Sigma[t] = exp( s[t]);
        ad[t] = -Sigma[t] * Hbb_1[t];
        m[t] = Sigma[t] * (cbb[t] - Hbb_1[t-1] * m[t-1]);
    }
}

static void alC_guess( State_parameter *theta_alpha, State *state )
{
    int t, n = state->n;
    double *alC = state->alC;
    double *d_tm = theta_alpha->d_tm, *phi_tm = theta_alpha->phi_tm;
    if( theta_alpha->is_basic ) {
        alC[n-1] = state->m_prior[n-1];
        for( t=n-2; t>=0; t-- )
            alC[t] = state->ad_prior[t] * alC[t+1] + state->m_prior[t];
    }
    else {
        alC[0] = theta_alpha->d_tm[0];
        for( t=1; t<n; t++ )
            alC[t] = d_tm[t] + phi_tm[t] * alC[t-1];
    }
}

static
void alC_pass_safe( State *state, double *inf_norm_distance )
{
    double lambda = 2.0; // Maximum multiple of prior standard deviation
    double lambda2 = lambda * lambda;
    int t, n = state->n;
    double *alC = state->alC, *m = state->m, *ad = state->ad;
    
    verify(isfinite(alC[n-1]),
    "Value of x_mode_t is infinite or not a number", "t=%d, n=%d", n-1, n);
    state->fatal_error_detected = !isfinite(alC[n-1]);
    
    double diff = m[n-1] - alC[n-1];
    
    if( diff*diff > lambda2 * state->Sigma[n-1] )
        diff *= lambda * sqrt(state->Sigma_prior[n-1]) / fabs(diff);
    
    *inf_norm_distance = fabs(diff);
    alC[n-1] += diff;
    
    for( t=n-2; t>=0; t-- )
    {
        verify(isfinite(alC[t]),
        "Value of x_mode_t is infinite or not a number", "t=%d, n=%d", t, n);
        state->fatal_error_detected = !isfinite(alC[t]);
        
        double old_alC_t = alC[t];
        alC[t] = m[t] + ad[t] * alC[t+1];
        diff = alC[t] - old_alC_t;
        
        if( diff*diff > lambda2 * state->Sigma_prior[t] )
        {
            diff *= lambda * sqrt(state->Sigma_prior[t]);
            alC[t] = old_alC_t + diff;
        }
        if( fabs(diff) > *inf_norm_distance )
        *inf_norm_distance = fabs(diff);
    }
    verify(isfinite(*inf_norm_distance),
    "Value of inf_norm_distance is infinite or not a number", "t=%d, n=%d", t, n);
    state->fatal_error_detected = !isfinite(*inf_norm_distance);
}

static 
void alC_pass( State *state, double *inf_norm_distance, int use_add )
{
	int t, n = state->n;
	double *alC = state->alC, *m = state->m, *ad = state->ad, *add = state->add;
    double diff = m[n-1] - alC[n-1];
    
    *inf_norm_distance = fabs(diff);
    alC[n-1] = m[n-1];
    
    for( t=n-2; t>=0; t-- ) {
   	double old_alC_t = alC[t];
   	alC[t] = m[t] + ad[t] * alC[t+1];
        
		if( use_add )
			alC[t] += 0.5 * add[t] * diff * diff;
			
      diff = alC[t] - old_alC_t;
        
        if( isnan(diff) ) {
			*inf_norm_distance = diff;
			return;
		}
        if( fabs(diff) > *inf_norm_distance )
            *inf_norm_distance = fabs(diff);
    }
}

static 
void compute_derivatives( Observation_model *model, State *state, int ad_add_only )
{
    int t, n = state->n, stride = state->psi_stride;
	double *alC = state->alC;
	double *ad = state->ad, *add = state->add, *addd = state->addd, *adddd = state->adddd;
	double *b = state->b, *bd = state->bd, *bdd = state->bdd, *bddd = state->bddd;
	double *mu = state->mu, *mud = state->mud, *mudd = state->mudd;
	double *sd = state->sd, *sdd = state->sdd, *sddd = state->sddd;
	double *Sigma = state->Sigma;
	double *Hbb_1 = state->Hbb_1;
	int n_alpha_partials = state->n_alpha_partials;
	
	if( ad_add_only )
		n_alpha_partials = 2;
    
	double add_prev = 0.0, addd_prev = 0.0, adddd_prev = 0.0;
	double mu_a_prev = 0.0, mud_ad_prev = 0.0, mudd_add_prev = 0.0;
	double bddd_addd_prev = 0.0;
	
    for( t=0; t<n-1; t++ )
    {
		double *psi_t = state->psi + t * stride;
		double gamma_t = (t==0) ? 0.0 : -Hbb_1[t-1] * Sigma[t];
		ad[t] = -Hbb_1[t] * Sigma[t];
		double ad_t_2 = ad[t] * ad[t];
		double ad_t_3 = ad_t_2 * ad[t];
        
		if( n_alpha_partials >= 2 ) 
		{
			double Z_1 = Sigma[t] * psi_t[3] + gamma_t * add_prev;
			sd[t] = Z_1 * ad[t];
			add[t] = Z_1 * ad_t_2;
			if( n_alpha_partials >= 3 )
			{
				double Z_2 = Sigma[t] * psi_t[4] + gamma_t * addd_prev;
				double Z_1_2 = Z_1 * Z_1;
				sdd[t] = (Z_2 + 2 * Z_1_2) * ad_t_2;
				addd[t] = (Z_2 + 3 * Z_1_2) * ad_t_3;
				if( n_alpha_partials >= 4 )
				{
					double ad_t_4 = ad_t_3 * ad[t];
					double Z_3 = Sigma[t] * psi_t[5] + gamma_t * adddd_prev;
					double Z_1_Z_2 = Z_1 * Z_2;
					double Z_1_3 = Z_1_2 * Z_1;
					sddd[t] = (Z_3 + 7 * Z_1_Z_2 + 8 * Z_1_3) * ad_t_3;
					adddd[t] = (Z_3 + 10 * Z_1_Z_2 + 15 * Z_1_3) * ad_t_4;
					adddd_prev = adddd[t];
				}
				addd_prev = addd[t];
			}
			add_prev = add[t];
		}
        
		if( !ad_add_only ) {
			double gamma_mu_a_prev = gamma_t * mu_a_prev;
			double gamma_mud_ad_prev = gamma_t * mud_ad_prev;
			double gamma_mudd_add_prev = gamma_t * mudd_add_prev;
			double gamma_muddd_addd_prev = gamma_t * bddd_addd_prev;
			
			double discount1 = 1.0 / (1.0 - gamma_mud_ad_prev);
			double discount2 = discount1 * discount1;
			double discount3 = discount2 * discount1;
			double discount4 = discount3 * discount1;
			
			double sd_t_2 = sd[t] * sd[t];
			double sd_t_3 = sd[t] * sd_t_2;
			
			double D1 = -sd[t] - gamma_mudd_add_prev * ad[t];
			double D2 = sd_t_2 - sdd[t]
				- gamma_mudd_add_prev * add[t]
				- gamma_muddd_addd_prev * ad_t_2;
			double D3 = -sddd[t] + 3*sd[t]*sdd[t] - sd_t_3
            - gamma_mudd_add_prev * addd[t]
            - 3 * gamma_muddd_addd_prev * ad[t] * add[t];
			
			double N0 = gamma_mu_a_prev;
			double N1 = gamma_mud_ad_prev * ad[t];
			double N2 = gamma_mud_ad_prev * add[t] + gamma_mudd_add_prev * ad_t_2;
			double N3 = gamma_mud_ad_prev * addd[t]
            + 3 * gamma_mudd_add_prev * ad[t] * add[2]
				+ gamma_muddd_addd_prev * ad_t_3;
			
			double V0 = discount1;
			double V1 = -D1*discount2;
			double V2 = -D2*discount2 + 2*D1*D1*discount3;
			double V3 = -D3*discount2 + 6*D1*D2*discount3 - 6*D1*D1*D1*discount4;
			
			double b_a = N0*V0;
			double bd_ad = N1*V0 + N0*V1;
			double bdd_add = N2*V0 + 2*N1*V1 + N0*V2;
			double bddd_addd = N3*V0 + 3*N2*V1 + 3*N1*V2 + N0*V3;
			
			/* Compute b and derivatives */
			b[t] = alC[t] + b_a;
			bd[t] = ad[t] + bd_ad;
			bdd[t] = add[t] + bdd_add;
			bddd[t] = addd[t] + bddd_addd;
			
			double den_factor = -1/Hbb_1[t];
			double bx_1 = 1.0 / bd[t];
			double b2_1 = bdd[t] * bx_1;
			double b22_11 = b2_1 * b2_1;
			double b3_1 = bddd[t] * bx_1;
			double b4_1 = adddd[t] * bx_1;
			
			double mu_b = 0.5*b2_1*den_factor;
			double mud_bd = 0.5*(b3_1-b22_11)*den_factor;
			double mudd_bdd = 0.5*(b4_1-3*b2_1*b3_1+2*b2_1*b22_11)*den_factor;
			
			mu[t] = b[t] + mu_b;
			mud[t] = bd[t] + mud_bd;
			mudd[t] = bdd[t] + mudd_bdd;
			
			mu_a_prev = b_a + mu_b;
			mud_ad_prev = bd_ad + mud_bd;
			mudd_add_prev = bdd_add + mudd_bdd;
			bddd_addd_prev = bddd_addd;
		}
	}
	
	double gamma_t = -Hbb_1[n-2] * Sigma[n-1];
	double den_factor = 1.0 / (1.0 - gamma_t * mud_ad_prev);
	double bn_an = gamma_t * mu_a_prev * den_factor; 
	b[n-1] = alC[n-1] + bn_an;
    mu[n-1] = b[n-1];
	
	bd[n-1] = bdd[n-1] = bddd[n-1] = 0.0;
	ad[n-1] = add[n-1] = addd[n-1] = adddd[n-1] = 0.0;
}

static 
int compute_alC( int trust_alC, int safe, Observation_model *model, Theta *theta, State *state, Data *data )
{
    int iteration;
    double inf_norm_distance;
    
	if( !trust_alC )
        alC_guess( theta->alpha, state );
    
    int max_iterations = safe ? 5 * state->max_iterations : state->max_iterations;
    
    // Iterate to find mode
    for( iteration = 0; iteration < max_iterations; iteration++ ) 
    {
        make_Hbb_cbb( model, theta, state, data );
        compute_Sigma_m( state );
        compute_derivatives( model, state, TRUE );
        
        if( safe )
            alC_pass_safe( state, &inf_norm_distance );
        else
            alC_pass( state, &inf_norm_distance, TRUE );
        
        if( inf_norm_distance < state->tolerance )
            return 1;	// Success, convergence to required tolerance achieved
        if( isnan(inf_norm_distance) || isinf(inf_norm_distance) )
            return 0; // Failure, not a number error somewhere
    }
    return 0; // Failure, non-convergence in given maximum number of iterations
}

void compute_alC_all( Observation_model *model, Theta *theta, State *state, Data *data )
{
    make_Hb_cb( theta->alpha, state );
    compute_prior_Sigma_ad( state );
    
    if( !state->guess_alC )
        alC_guess(theta->alpha, state);
    
    // At first trust_alC = TRUE by default
    if( !compute_alC( TRUE, FALSE, model, theta, state, data ) )
        compute_alC( FALSE, TRUE, model, theta, state, data );
    
    // // Check fatal error (no error checking during compute_alC(...))
    // if( state->fatal_error_detected ) return;

    make_Hbb_cbb( model, theta, state, data );
    compute_Sigma_m_plus( state );
    compute_derivatives( model, state, FALSE );
}

void draw_HESSIAN(int isDraw, Observation_model *model,Theta *theta, State *state, Data *data, double *log_q )
{
    int t, n = state->n;
    double *alpha = state->alpha;
    double *alC = state->alC;
    double *b = state->b, *bd = state->bd, *bdd = state->bdd, *bddd = state->bddd;
    double *mu = state->mu, *mud = state->mud, *mudd = state->mudd;
    double *adddd = state->adddd;
    double *Hb_0 = state->Hb_0, *Hb_1 = state->Hb_1, *cb = state->cb;
    double psi_b_t[6];
    double *Sigma_prior = state->Sigma_prior;
    
    double threshold = 0.1;
    double K_1_threshold;
    double K_2_threshold[6];
    const int max_n_reject = 100;
    
    
    /* Precomputation */
    Symmetric_Hermite *sh = symmetric_Hermite_alloc(100, 100); // Maximum value of K, maximum number of rejection sampling rejections
    K_1_threshold = sqrt(12.0 * threshold);
    K_2_threshold[2] = sqrt(2.0 * threshold);
    K_2_threshold[3] = exp( log(6.0 * threshold)/3.0 );
    K_2_threshold[4] = exp( log(24.0 * threshold)/24.0 );
    K_2_threshold[5] = exp( log(120.0 * threshold)/120.0 );   
    
    *log_q = 0.0;
    double d1 = 0.0;
    for( t=n-1; t>=0; t-- )
    {
        
        /* Compute bttp */
        double d2_2 = 0.5 * d1 * d1;
        double d3_6 = (1.0/3) * d2_2 * d1;
        double d4_24 = 0.25 * d3_6 * d1;
        double bttp = b[t] + bd[t]*d1 + bdd[t]*d2_2 + bddd[t]*d3_6 + adddd[t]*d4_24;
        
        
        /* Compute mutmt and derivatives at bttp */
        double D1 = bttp - alC[t];
        double D2_2 = 0.5 * D1 * D1;
        double D3_6 = (1.0/3) * D2_2 * D1;
        double D4_24 = 0.25 * D3_6 * D1;
        double mutmt, mutmt1, mutmt2, mutmt3, mutmt4;
        
        if (t>0)
        {
            mutmt = mu[t-1] + mud[t-1]*D1 + mudd[t-1]*D2_2 + bddd[t-1]*D3_6 + adddd[t-1]*D4_24;
            mutmt1 = mud[t-1] + mudd[t-1]*D1 + bddd[t-1]*D2_2 + adddd[t-1]*D3_6;
            mutmt2 = mudd[t-1] + bddd[t-1]*D1 + adddd[t-1]*D2_2;
            mutmt3 = bddd[t-1] + adddd[t-1]*D1;
            mutmt4 = adddd[t-1];
        }
        
        
        /* Compute psi_t and derivatives and h1, h2, h3, h4, h5 at bttp */
        model->compute_derivatives_t( theta, data, t, bttp, psi_b_t );		
        double h1 = cb[t] - Hb_0[t]*bttp + psi_b_t[1];
        
        if( t<n-1 )
            h1 -= Hb_1[t] * alpha[t+1];
        
        /* Initialize skew */
        Skew_parameters skew;
        skew.h2 = -Hb_0[t] + psi_b_t[2];
        skew.h3 = psi_b_t[3];
        skew.h4 = psi_b_t[4];
        skew.h5 = psi_b_t[5];
        
        if( t>0 )
        {
            h1 -= Hb_1[t-1]*mutmt;
            skew.h2 -= Hb_1[t-1]*mutmt1;
            skew.h3 -= Hb_1[t-1]*mutmt2;
            skew.h4 -= Hb_1[t-1]*mutmt3;
            skew.h5 -= Hb_1[t-1]*mutmt4;
        }
        
        skew.mode = bttp - h1/skew.h2;
        skew.s2_prior = Sigma_prior[t];
        skew.u_sign = state->sign;
        skew.is_draw = isDraw;
        skew.n_reject = 0;  // SG: need to be initialize for verify(..)
        skew.z = alpha[t];
        
        /* Draw/Eval */
        skew_draw_eval( &skew, sh, K_1_threshold, K_2_threshold );

        // Compute diagnostics
        if( state->compute_diagnostics ) compute_diagnostics(t, state, &skew);
        
        
        verify(skew.n_reject < max_n_reject,
        "Maximum number of skew rejects exceeded", "t=%d, n=%d", t, n);
        state->fatal_error_detected = !(skew.n_reject < max_n_reject);
        if( state->fatal_error_detected ) return;

        verify(isfinite(skew.z),
        "Draw of x_t is infinite or not a number", "t=%d, n=%d", t, n);
        state->fatal_error_detected = !isfinite(skew.z);
        if( state->fatal_error_detected ) return;

        verify(isfinite(skew.log_density),
        "Log density of x_t is infinite or not a number", "t=%d, n=%d", t, n);
        state->fatal_error_detected = !isfinite(skew.log_density);
        if( state->fatal_error_detected ) return;

        

        alpha[t] = skew.z;
        
        *log_q += skew.log_density;
        d1 = alpha[t] - alC[t];
        
        state->a[t] = d1;
        state->eps[t] = alpha[t] - bttp;
	}
}
