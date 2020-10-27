#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "RNG.h"
#include "skew.h"
#include "skew_spline.h"
#include "symmetric_Hermite.h"

void skew_draw_parameter_string(Skew_parameters *skew, char *msg)
{
    double sigma2 = -1/skew->h2;
    double sigma = sqrt(sigma2);
    double sigma3 = sigma*sigma2;
    double sigma4 = sigma*sigma3;
    double sigma5 = sigma*sigma4;
    sprintf(msg, "\nh2\t%lf\nh3\t%lf\t%lf\nh4\t%lf\t%lf\nh5\t%lf\t%lf\n",
            skew->h2, skew->h3, skew->h3*sigma3, skew->h4, skew->h4*sigma4, skew->h5, skew->h5*sigma5);
}

#define skew_draw_K 20

void skew_draw_eval(Skew_parameters *skew, Symmetric_Hermite *sh,
                    double K_1_threshold, double *K_2_threshold)
{
    int i, j;   // Indices

  if (skew->s2_prior > 0.5) {
    double skew_h[6];
    double omega = 1.0 / skew->s2_prior;
    double mu = 0.0; // unused
    int K = 20;
    skew_h[2] = skew->h2 + omega;
    skew_h[3] = skew->h3;
    skew_h[4] = skew->h4;
    skew_h[5] = skew->h5;
    int is_v = (skew->s2_prior * skew->h2 + 1.0) < -4.0;
    skew_spline_draw_eval(skew->is_draw, K, is_v, skew->mode, skew_h, mu, omega,
                          &(skew->z), &(skew->log_density));
    return;
  }

    // Set secondary parameters
    double sigma_m2 = -skew->h2;
    double sigma_2 = 1.0 / sigma_m2;
    double sigma = sqrt(sigma_2);
    double sigma_3 = sigma_2 * sigma;
    double a3 = skew->h3 * sigma_3 / 6.0;
    double a4 = skew->h4 * sigma_2 * sigma_2 / 24.0;
    double a5 = skew->h5 * sigma_3 * sigma_2 / 120.0;

    double z_bar = 5.0;                     // Original parameter
    double prior_pi = 1e-9;                 // Original parameter
    double x_bar = z_bar * sigma;
    double prior_sigma = sqrt(skew->s2_prior);

    double w_1 = fabs(125*a3 + 3125*a5);
    int K_1 = (w_1 > K_1_threshold) ? 2 : 1;
    double w_2 = fabs(625*a4);
    int K_2 = 1;

    for( i=2; i<=5; i++ )
        if( w_2 > K_2_threshold[i] )
            K_2 = i;
    if( (a4 < 0.0) && (K_2 % 2) )
        K_2++;

    Skew_Approximation skew_approx;
    skew_approx.n1 = 5 * K_1;
    skew_approx.n2 = 2 * K_2;

    // Compute polynomial in y=x^2 approximating cosh(a3 x^3 + a5 x^5)
    double a33 = a3*a3;
    double a35 = a3*a5;
    double a55 = a5*a5;
    skew_approx.p1[0] = 1.0;
    skew_approx.p1[1] = skew_approx.p1[2] = 0.0;
    skew_approx.p1[3] = 0.5 * a33;
    skew_approx.p1[4] = a35;
    skew_approx.p1[5] = 0.5 * a55;
    if (K_1 > 1) {
        skew_approx.p1[6] = a33*a33/24;
        skew_approx.p1[7] = a33*a35/6;
        skew_approx.p1[8] = 0.25 * a35*a35;
        skew_approx.p1[9] = a35*a55/6;
        skew_approx.p1[10] = a55*a55/24;
    }

    // Compute polynomial in y=x^2 approximating exp(a4 x^4)
    skew_approx.p2[0] = 1.0;
    for (i=1; i<=K_2; i++) {
        skew_approx.p2[2*i] = skew_approx.p2[2*(i-1)] * a4 / i;
        skew_approx.p2[2*i-1] = 0.0;
    }

    // Compute coefficients of y=x^2 in product polynomial
    for (j=0; j<=skew_approx.n2; j++)
        sh->coeff[j] = skew_approx.p2[j];
    for (; j<=skew_approx.n1 + skew_approx.n2; j++)
        sh->coeff[j] = 0.0;
    for(i=3; i<=skew_approx.n1; i++) {
        sh->coeff[i] += skew_approx.p1[i];
        for(j=1; j<=skew_approx.n2; j++)
            sh->coeff[i+j] += skew_approx.p1[i] * skew_approx.p2[j];
    }
    sh->K = skew_approx.n1 + skew_approx.n2 + 1;

    double x_star, z_star, y, log_density;
    int tail_draw;
    if (skew->is_draw) { // Draw x_star and evaluate sigma density

        // Draw from g_1 or g_2 according to weights (1-prior_pi), prior_pi
        double U_component = rng_rand();
        if (U_component < prior_pi) {
            y = rng_chi2( 3.0 );
            x_star = x_bar + prior_sigma * sqrt(y);
        if (U_component < 0.5 * prior_pi)
                x_star = -x_star;
            z_star = x_star / sigma;
            log_density = symmetric_Hermite_log_f(sh, z_star);
        }
        else {
            symmetric_Hermite_draw(sh, &z_star, &log_density);
            skew->n_reject = sh->n_reject;
            z_star *= skew->u_sign;
            x_star = z_star * sigma;
        }
        tail_draw = (fabs(x_star) > x_bar);

        // Compute f(x)
        double term_2 = (tail_draw) ? (z_bar * z_bar) : (z_star * z_star);
        double term_4 = term_2 * term_2;
        double f_x = (a3 * term_2 + a5 * term_4) * z_star;
        double exp_term = exp(2.0 * f_x);

        // Random change of sign
        if (rng_rand() * (1.0 + exp_term) < (1.0 - exp_term)) {
            x_star = -x_star;
            z_star = -z_star;
            log_density -= log(0.5 + 0.5*exp_term);
        }
        else
            log_density += 2.0 * f_x - log(0.5 + 0.5*exp_term);
        skew->z = x_star + skew->mode;
    }
    else { // Just evaluate log density without drawing
        x_star = skew->z - skew->mode;
        z_star = x_star / sigma;
        tail_draw = (fabs(x_star) > x_bar);

        // Compute f(x)
        double term_2 = (tail_draw) ? (z_bar * z_bar) : (z_star * z_star);
        double term_4 = term_2 * term_2;
        double f_x = (a3 * term_2 + a5 * term_4) * z_star;
        log_density = symmetric_Hermite_log_f(sh, z_star);
        log_density -= log(0.5 + 0.5*exp(-2.0*f_x));
    }

    log_density += log1p(-prior_pi) - log(sigma);
    if (tail_draw) {
        y = (fabs(z_star) - z_bar) * ( fabs(z_star) - z_bar );
        double tail_density = rng_chi2_pdf( y, 3.0 ) / prior_sigma;
        log_density = log( exp(log_density) + prior_pi*tail_density );
    }
    skew->log_density = log_density;
}
