#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "alias.h"
#include "skew_spline.h"
#include "skew_grid.h"
#include "RNG.h"
#include "Phi.h"
#include "spline.h"

#include "mex.h"

#define max_h 5
#define max_K 22

// Parameters
const static double c_k = 0.5;   // Threshold for change in 2nd derivative
const static double c_p = 0.002; // Minimal last value as fraction of previous
const static double c_m = 0.002; // Minimal last slope as fraction of previous
const static double phi_o_lin_trans = 5.0;
const static double phi_o_tanh_max = 5.0;

// Used options(digits=22), log(sqrt(2/pi)) in R
const static double log_2_pi = 1.837877066409345339082;
const static double p_0 = 2.0;
const static double m_0K = -2.0; // Gives m_0 when m_0K/K normalization is computed.
const static int n_powers = 5;
extern Grid *grids[];
extern int min_grid_points, max_grid_points;

static int K, Kp1, k_bar_plus, k_bar_minus;

// These quantities depend on the current target parameters, so cannot be
// precomputed. They have values at all grid points.
// ----------------------------------------------------------------------

// Level p of spline density at grid points and component probabilities of
// spline density
static double p[max_K];   // Spline values (levels)
static double pi[max_K];  // Component probabilities

// If phi(z) is a function approximating log f(y|z), then
// phi+(z) = phi(z) for z>=0, and phi-(z) = phi(-z) for z>=0.
// The next two vectors are values of these functions on a grid of (positive)
// values of z.
static double phi_plus[max_K];  // a2 x^2/2 + a3 x^3/6 + a4 x^4/24 + a5 x^5/120
static double phi_minus[max_K]; // a2 x^2/2 - a3 x^3/6 + a4 x^4/24 - a5 x^5/120

// Other evaluations
static double exp_phi_plus[max_K];
static double exp_phi_minus[max_K];
static double r[max_K];

// Functions related to F_v(v) = 1-(1-v)^delta, for delta=0.5
static inline double inverse_F_v(double u) {return 1.0-(1.0-u)*(1.0-u);}
static inline double F_v(double v) {return 1.0 - sqrt(1.0-v);}
static inline double f_v(double v) {return 0.5/sqrt(1.0-v);}
static inline double ln_f_v(double v) {return -M_LN2 - 0.5*log(1.0-v);}

#ifndef max
static inline double max(double a, double b) {return (a>b) ? a : b;}
static inline double min(double a, double b) {return (a>b) ? b : a;}
#endif

#ifndef true
#define true (1)
#define false (0)
#endif

static double sigma[max_h + 1], a[max_h + 1];
static double ae_bar[max_h + 1], ao_bar[max_h + 1];
static double a_bar_plus[4], a_bar_minus[4];         // Cubic approximation
static double a_bar_bar_plus[2], a_bar_bar_minus[2]; // Linear approximation
static double x_linear_plus = DBL_MAX, x_linear_minus = DBL_MAX;
static double m_0, m_K, m_Km1;
static double *x1, *x2, *x3, *x4, *x5;

static inline double cubic_Taylor(double *d, double x)
{
  return (( (d[3]/6)*x + d[2]/2 )*x + d[1])*x + d[0];
}

static inline double quad_Taylor(double *d, double x)
{
  return ( (d[2]/2)*x + d[1] )*x + d[0];
}

static inline double linear_Taylor(double *d, double x)
{
  return d[1]*x + d[0];
}

// Compute m_k. Needs to be called only for knot K-1, knots k and k+1
// satifying x_k < |x| < x_{k+1}
static inline double compute_m_k(int k, Grid *g, int is_v, int is_trunc)
{
  // For k==0, m[k] is a known value. Quickly return.
  if (k==0) return is_v ? m_0 : 0.0;

  // Compute derivatives phi_plus_prime/phi_minus_prime of phi_plus/phi_minus
  double phi_even_prime, phi_odd_prime, phi_plus_prime, phi_minus_prime;
  int cubic_plus = (k_bar_plus > 0) && (k > k_bar_plus);
  int cubic_minus = (k_bar_minus > 0) && (k > k_bar_minus);

  if (!cubic_plus || !cubic_minus) {
    phi_even_prime = a[2] * x1[k] + a[4] * x3[k];
    phi_odd_prime = a[3] * x2[k] + a[5] * x4[k];
  }

  if (cubic_plus) {
    double delta = (x1[k] - x1[k_bar_plus]);
    if (delta < x_linear_plus)
      phi_plus_prime = quad_Taylor(a_bar_plus+1, delta);
    else
      phi_plus_prime = a_bar_bar_plus[1];
  } else
    phi_plus_prime = phi_even_prime + phi_odd_prime;

  if (cubic_minus) {
    double delta = (x1[k] - x1[k_bar_minus]);
    if (delta < x_linear_minus)
      phi_minus_prime = quad_Taylor(a_bar_minus+1, delta);
    else
      phi_minus_prime = a_bar_bar_minus[1];
  } else
    phi_minus_prime = phi_even_prime - phi_odd_prime;

  double r_prime = 0.5*(exp_phi_plus[k] * phi_plus_prime
                        +exp_phi_minus[k] * phi_minus_prime);

  // Compute m_k
  double m_k;
  if (is_v) {
    m_k = r_prime * g->c[k] - p[k] * g->f_v_prime[k];
    m_k *= g->Kf_v2_inv[k];
  }
  else {
    m_k = r_prime * g->cu[k];
    m_k *= g->K_inv;
  }

  // |m_k| > 3.0 p_k is infeasible in spline draw, return feasible value if
  // is_trunc is true.
  if (is_trunc && (fabs(m_k)/p[k] > 3.0))
    m_k /= fabs(m_k)/p[k];
  return m_k;
}

// Compute odd part of phi function
static inline double phi_odd(double x, double *a)
{
  double abs_x = fabs(x), phi_o, phi_e;
  double phi_plus_abs_x, phi_minus_abs_x;
  int cubic_plus = (k_bar_plus > 0) && (abs_x > x1[k_bar_plus]);
  int cubic_minus = (k_bar_minus > 0) && (abs_x > x1[k_bar_minus]);

  if (!cubic_plus || !cubic_minus) {
    double x_sq = x*x;
    double x_cubed = x_sq * abs_x;
    phi_o = ((a[5]/120 * x_sq) + a[3]/6) * x_cubed;
    phi_e = ((a[4]/24 * x_sq) + a[2]/2) * x_sq;
    phi_plus_abs_x = phi_e + phi_o;
    phi_minus_abs_x = phi_e - phi_o;
  }

  if (cubic_plus) {
    double delta = abs_x - x1[k_bar_plus];
    if (delta > x_linear_plus)
      phi_plus_abs_x = linear_Taylor(a_bar_bar_plus, delta - x_linear_plus);
    else
      phi_plus_abs_x = cubic_Taylor(a_bar_plus, delta);
  }

  if (cubic_minus) {
    double delta = abs_x - x1[k_bar_minus];
    if (delta > x_linear_minus)
      phi_minus_abs_x = linear_Taylor(a_bar_bar_minus, delta - x_linear_minus);
    else
      phi_minus_abs_x = cubic_Taylor(a_bar_minus, delta);
  }
  phi_o = 0.5 * (phi_plus_abs_x - phi_minus_abs_x);
  if (x < 0.0)
    phi_o = -phi_o;

  if (phi_o > phi_o_lin_trans)
    phi_o = phi_o_lin_trans
      + phi_o_tanh_max * tanh((phi_o - phi_o_lin_trans) / phi_o_tanh_max);
  if (phi_o < -phi_o_lin_trans)
    phi_o = -phi_o_lin_trans
      + phi_o_tanh_max * tanh((phi_o + phi_o_lin_trans) / phi_o_tanh_max);
  return phi_o;
}

/* If is_draw is true, draw from the skew_draw distribution and evaluate
 * the log density of the draw. If not, evaluate at the point *z
 *
 * Inputs:
 *  is_draw:       whether to draw or not
 *  n_grid_points: number of grid points in precomputed grid
 *  is_v:          whether or not to do v = F_v^{-1}(u) transformation
 *  mode:          mode of the skew_draw distribution
 *  h:             vector of derivatives (h[2] - h[5] used) of phi at mode
 *  mu:            prior mean
 *  omega:         prior precision
 *
 * Input or output according to value of id_draw:
 *  z:             pointer to value of draw/point of evaluation
 *
 * Outputs:
 *  ln_f:          pointer to log density evaluation
 */
void skew_spline_draw_eval(int is_draw, int n_grid_points, int is_v,
                           double mode, double *h, double mu, double omega,
                           double *z, double *ln_f)
{
  // Precomputed values on specified grid
  Grid *g = grids[n_grid_points];

  int i, k;           // Using i for powers, k for knots
  double x, v, u, t;  // x is z/sigma = z*sqrt(omega)

  K = g->K;
  Kp1 = K+1;
  m_0 = is_v ? (m_0K * g->K_inv) : 0.0;

  // Compute powers of sigma, the prior standard deviation
  sigma[2] = 1.0/omega;
  sigma[1] = sqrt(sigma[2]);
  sigma[3] = sigma[1] * sigma[2];
  sigma[4] = sigma[2] * sigma[2];
  sigma[5] = sigma[3] * sigma[2];

  // Compute prior-precision-normalized coefficients a[2] through a[5]
  // Note that these give the same value:
  //   a[2] x^2/2 + a[3] x^3/6 + a[4] x^4/24 + a[5] x^5/120
  //   h[2] z^2/2 + h[3] z^3/6 + h[4] z^4/24 + h[5] z^5/120
  for (i=2; i<=5; i++)
    a[i] = sigma[i] * h[i];

  // Evalute level p[k] = f_even at all grid points k = 0, 1, ..., K
  double *x_plus = is_v ? g->x_plus : g->xu_plus;
  x1 = x_plus + 1*Kp1;
  x2 = x_plus + 2*Kp1;
  x3 = x_plus + 3*Kp1;
  x4 = x_plus + 4*Kp1;
  x5 = x_plus + 5*Kp1;

  k_bar_plus = -1;
  k_bar_minus = -1;
  for (k=1; k<=K; k++) {
    double even_term, odd_term;
    if (k_bar_plus < 0 || k_bar_minus < 0) {
      even_term = a[2] * x2[k] + a[4] * x4[k];
      odd_term = a[3] * x3[k] + a[5] * x5[k];
    }

    if (k_bar_plus < 0)
      phi_plus[k] = even_term + odd_term;
    else {
      double delta = x1[k] - x1[k_bar_plus];
      if (delta > x_linear_plus)
        phi_plus[k] = linear_Taylor(a_bar_bar_plus, delta - x_linear_plus);
      else
        phi_plus[k] = cubic_Taylor(a_bar_plus, delta);
    }

    if (k_bar_minus < 0)
      phi_minus[k] = even_term - odd_term;
    else {
      double delta = x1[k] - x1[k_bar_minus];
      if (delta > x_linear_minus)
        phi_minus[k] = linear_Taylor(a_bar_bar_minus, delta - x_linear_minus);
      else
        phi_minus[k] = cubic_Taylor(a_bar_minus, delta);
    }

    exp_phi_plus[k] = exp(phi_plus[k]);
    exp_phi_minus[k] = exp(phi_minus[k]);
    p[k] = r[k] = 0.5 * (exp_phi_plus[k] + exp_phi_minus[k]);
    if (is_v) p[k] *= g->f_v_inv[k];

    double d2_e = a[2] + a[4] * x2[k];
    double d2_o = a[3] * x1[k] + a[5] * x3[k];
    if (k_bar_plus < 0 && d2_e + d2_o > c_k * a[2])
      k_bar_plus = k;
    if (k_bar_minus < 0 && d2_e - d2_o > c_k * a[2])
      k_bar_minus = k;
    if (k_bar_minus == k || k_bar_plus == k) {
      ae_bar[0] = a[2]*x2[k] + a[4]*x4[k];
      ao_bar[0] = a[3]*x3[k] + a[5]*x5[k];
      ae_bar[1] = a[2]*x1[k] + a[4]*x3[k];
      ao_bar[1] = a[3]*x2[k] + a[5]*x4[k];
      ae_bar[2] = a[2]       + a[4]*x2[k];
      ao_bar[2] = a[3]*x1[k] + a[5]*x3[k];
      ae_bar[3] =              a[4]*x1[k];
      ao_bar[3] = a[3]       + a[5]*x2[k];
    }
    if (k_bar_plus == k) {
      for (i=0; i<=3; i++)
        a_bar_plus[i] = ae_bar[i] + ao_bar[i];
      if (a_bar_plus[3] > 0.0) {
        x_linear_plus = fabs(a_bar_plus[2]/a_bar_plus[3]);
        a_bar_bar_plus[0] = cubic_Taylor(a_bar_plus, x_linear_plus);
        a_bar_bar_plus[1] = quad_Taylor(a_bar_plus+1, x_linear_plus);
      }
    }
    if (k_bar_minus == k) {
      for (i=0; i<=3; i++)
        a_bar_minus[i] = ae_bar[i] - ao_bar[i];
      if (a_bar_minus[3] > 0.0) {
        x_linear_minus = fabs(a_bar_minus[2]/a_bar_minus[3]);
        a_bar_bar_minus[0] = cubic_Taylor(a_bar_minus, x_linear_minus);
        a_bar_bar_minus[1] = quad_Taylor(a_bar_minus+1, x_linear_minus);
      }
    }
  }

  // Computation of p[K] and m[K] are based on true computed values
  double p_tau = p[K], m_tau = compute_m_k(K, g, is_v, false);
  m_Km1 = compute_m_k(K-1, g, is_v, false);
  if (is_v) {
    p[0] = p_0;
    p[K] = 0.0;
    m_K = min(c_m*m_tau, -6.0*p[K-1] - 4.0*m_tau - m_Km1);
  }
  else {
    p[0] = 1.0;
    p[K] = -4.0*p_tau + 5.0*p[K-1] + 2*m_tau + m_Km1;
    m_K = min(0.0, 24*(p[K-1]-p_tau) + 8.0*m_tau + 5.0*m_Km1);
    if (p[K] < 0.0) {
      double p_min = c_p*p_tau;
      m_K += 6.0*(p_min - p[K]);
      p[K] = p_min;
    }
    m_K = min(0.0, m_K);
  }

  // From here on we need to truncate to avoid negative density values.
  if (m_Km1 < -3.0*p[K-1])
    m_Km1 = -3.0*p[K-1];

  // Compute knot probabilities and normalization constant.
  double p_Delta = p_tau - 0.5*(p[K-1] + p[K]) - 0.125*(m_Km1 - m_K);
  double m_Delta = m_tau - 1.5*(p[K] - p[K-1]) + 0.25*(m_Km1 + m_K);
  if (p_Delta < 0.0) {
    p_Delta = 0.0;
    m_Delta = 0.0;
  }
  else if (fabs(m_Delta) > 3.0*p_Delta) {
    m_Delta *= 3.0*p_Delta/fabs(m_Delta);
  }

  double pi_total = (pi[0] = 0.5*p[0] + m_0/12); // Contribution of first knot
  for (k=1; k<K; k++)
    pi_total += (pi[k] = p[k]);
  pi_total += (pi[K] = 0.5*p[K] - m_K/12);       // Contribution of last knot
  pi_total += (pi[K+1] = 0.5*p_Delta);           // Contribution of tau knot

  if (*z == 16.8125) {
    mexPrintf("a[2] = %lf, a[3] = %lf, a[4] = %lf, a[5] = %lf\n",
           a[2], a[3], a[4], a[5]);
    mexPrintf("k_bar_plus = %d, k_bar_minus = %d\n", k_bar_plus, k_bar_minus);
    if (k_bar_plus > 0)
      mexPrintf("x[k_bar_plus] = (%lf, %lf, %lf), x_linear_plus = %lf\n",
             x1[k_bar_plus], x2[k_bar_plus], x3[k_bar_plus], x_linear_plus);
    if (k_bar_minus > 0)
      mexPrintf("x[k_bar_minus] = (%lf, %lf, %lf), x_linear_minus = %lf\n",
             x1[k_bar_minus], x2[k_bar_minus], x3[k_bar_minus], x_linear_minus);

    mexPrintf("ae_bar[0] = %lf, ae_bar[1] = %lf, "
             "ae_bar[2] = %lf, ae_bar[3] = %lf\n",
             ae_bar[0], ae_bar[1], ae_bar[2], ae_bar[3]);
    mexPrintf("ao_bar[0] = %lf, ao_bar[1] = %lf, ao_bar[2] = %lf, "
             "ao_bar[3] = %lf\n",
             ao_bar[0], ao_bar[1], ao_bar[2], ao_bar[3]);

    mexPrintf("a_bar_plus[0,1,2,3] = (%lf, %lf, %lf, %lf)\n",
           a_bar_plus[0], a_bar_plus[1], a_bar_plus[2], a_bar_plus[3]);
    mexPrintf("a_bar_minus[0,1,2,3] = (%lf, %lf, %lf, %lf)\n",
           a_bar_minus[0], a_bar_minus[1], a_bar_minus[2], a_bar_minus[3]);

    mexPrintf("a_bar_bar_plus[0,1] = (%lf, %lf)\n",
           a_bar_bar_plus[0], a_bar_bar_plus[1]);
    mexPrintf("a_bar_bar_minus[0,1] = (%lf, %lf)\n",
           a_bar_bar_minus[0], a_bar_bar_minus[1]);

    for (k=0; k<=K+1; k++) {
      double mk = (k==(K+1)) ? 0.0 : compute_m_k(k, g, is_v, 0);
      mexPrintf("k=%d, pi[k]=%12.10lf, p[k]=%12.10lf, m[k]=%12.10lf\n", k, pi[k], p[k], mk);
    }
    mexPrintf("pi_total=%lf\n", pi_total);

    mexPrintf("p[K-1] = %le, m[K-1] = %le\np_tau = %le, m_tau = %le\n"
             "p[K] = %le, m[K] = %le\np_Delta = %le, m_Delta = %le\n",
           p[K-1], m_Km1, p_tau, m_tau, p[K], m_K, p_Delta, m_Delta);
  }

  // Repeatable part of draw starts here, in case a loop is desired.

  // Odd part of phi function and probability that sign of x is current sign
  double phi_o, expm2phi_o, f_sign_recip;

  // Normalize and draw or compute knot index.
  if (is_draw)
    draw_discrete(K+2, pi, 1, &k);
  else {
    x = (*z - mode)/sigma[1];
    phi_o = phi_odd(x, a);
    expm2phi_o = exp(-2.0 * phi_o);
    f_sign_recip = (1.0 + expm2phi_o);
    x = fabs(x);
    v = (x==0) ? 0 : 2*Phi(x)-1;
    u = is_v ? F_v(v) : v;
    k = floor(K*u);
    t = K*u - k;
  }

  // Evaluate derivative of f_even at this knot.
  double m_k, m_kp1;
  if (k<K-1)
    m_k = compute_m_k(k, g, is_v, true);
  else
    m_k = (k==K-1) ? m_Km1 : m_K;

  // Draw u, generate v then z
  if (is_draw) {

    // Draw u
    if (k==0) {
      t = left_t_draw(p[0], m_k);
      m_kp1 = compute_m_k(1, g, is_v, true);
    }
    else if (k==K) {
      k = k-1;
      m_k = m_Km1;
      m_kp1 = m_K;
      t = right_t_draw(p[K], m_K);
    }
    else if (k==K+1) {
      k = k-2;
      m_k = m_Km1;
      m_kp1 = m_K;
      t = (inner_t_draw(p_Delta, m_Delta/2) + 1) / 2;
    }
    else {
      t = inner_t_draw(p[k], m_k);
      if (t<0) {
        t = t+1; k = k-1;
        m_kp1 = m_k;
        m_k = compute_m_k(k, g, is_v, true);
      }
      else {
        m_kp1 = (k==K-1) ? m_K : compute_m_k(k+1, g, is_v, true);
      }
    }
    u = (k+t)/K;
    v = is_v ? inverse_F_v(u) : u;
    x = inverse_Phi(0.5 + 0.5*v);
    phi_o = phi_odd(x, a);
    expm2phi_o = exp(-2.0 * phi_o);
    f_sign_recip = 1.0 + expm2phi_o;
    if (rng_rand() * f_sign_recip > 1.0) {
      x = -x;
      phi_o = -phi_o;
    }
    *z = x*sigma[1] + mode;
  }
  else {
    m_kp1 = (k==K-1) ? m_K
                     : ((k==K-2) ? m_Km1 : compute_m_k(k+1, g, is_v, true));
  }

  // Compute exact value of log approximate density
  // ----------------------------------------------

  // Compute f_u factor
  double f_u, c0, c1, c2, c3;
  c0 = p[k];          // Constant coefficient in subinterval spline
  c1 = m_k;           // Coefficient of t
  c2 = -3*c0 - 2*c1 + 3*p[k+1] - m_kp1; // Coefficient of t^2
  c3 = 2*c0 + c1 - 2*p[k+1] + m_kp1;    // Coefficient of t^3
  f_u = (((c3*t+c2)*t+c1)*t+c0);
  if (k==(K-1)) {
    double t_tilde = 2*t-1; // In [-1, 1], corresponding to [u[K-1], 1]
    double tt_abs = fabs(t_tilde);
    double tt_sign = (t_tilde > 0) ? 1.0 : -1.0;
    f_u += ((2*tt_abs - 3)*tt_abs*tt_abs + 1) * p_Delta;
    f_u += tt_sign * (((tt_abs - 2.0)*tt_abs + 1.0)*tt_abs) * 0.5 * m_Delta;
  }

  // Compute terms of log approximate density
  *ln_f = log(2.0 * f_u / (f_sign_recip * pi_total * sigma[1])) + g->log_K;
  if (is_v)
    *ln_f += ln_f_v(v);
  if (x<0 && is_draw)
    *ln_f += 2.0 * phi_o;
  *ln_f -= 0.5 * (log_2_pi + x*x);
}
