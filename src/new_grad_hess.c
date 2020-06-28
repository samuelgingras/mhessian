#include <math.h>
#include "mex.h"

// The following functions are utilities for operations on 2nd order polynomials.
// They give 2nd order polynomial results; higher order terms are dropped.
// The vector (poly[0], poly[1], poly[2]) represents the polynomial
//
//    poly[0] + poly[1] x + poly[2] x^2.

// polynomial assignment by element
static inline void poly_set(double *poly, double p0, double p1, double p2)
{
    poly[0] = p0; poly[1] = p1; poly[2] = p2;
}

// polynomial assignment
static inline void poly_copy(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] = poly1[i];
}

// polynomial scalar multiplication
static inline void poly_scalar_mult(double *poly, double c)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] *= c;
}

// polynomial addition
static inline void poly_add(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] += poly1[i];
}

// polynomial assignment with scalar multiplication
static inline void poly_set_scalar_mult(double *poly, double c, const double *poly1)
{
  int i;
  for (i=0; i<3; i++)
    poly[i] = c * poly1[i];
}

// polynomial addition with scalar multiplication
static inline void poly_add_scalar_mult(double *poly, double c, const double *poly1)
{
  int i;
  for (i=0; i<3; i++)
    poly[i] += c * poly1[i];
}

// polynomial subtraction
static inline void poly_subtract(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] -= poly1[i];
}

// full polynomial multiplication
static inline void poly_mult(double *poly, const double *poly1, const double *poly2)
{
    poly[0] = poly1[0]*poly2[0];
    poly[1] = poly1[0]*poly2[1] + poly1[1]*poly2[0];
    poly[2] = poly1[0]*poly2[2] + poly1[1]*poly2[1] + poly1[2]*poly2[0];
}

static function set_Qmat(Qmat *Q, double coeff_11, double coeff_tt, double coeff_ttp)
{
    Q->coeff_11 = coeff_11;
    Q->coeff_tt = coeff_tt;
    Q->coeff_ttp = coeff_ttp;
}

static function set_qvec(qvec *q, double coeff_1, double coeff_t)
{
    q->coeff_1 = coeff_1;
    q->coeff_t = coeff_t;
}

void compute_grad_Hess(
    const mxArray *mxState,
    int n,
    double *mu,
    double phi,
    double omega,
    // Output
    double *grad,
    double *Hess,
    double *var
    )
{
    double mu[3], b[3], delta[3], deltaSigma[3], delta2[3], Sigma2[3];
    double E1[3], E2[3], C11[3], V1[3], V2[3];
    double bV1[3], b2V1[3];
    Qmat Q, Q2, Q22;
    qvec q, q2;

    // Set elements (1, 1), (t, t), (t, t+1) of the matrices Q, Q_2 and Q_{22}
    set_Qmat(Q,   1.0, 1+phi*phi,                   -phi);
    set_Qmat(Q2,  0.0, 2*phi*(1-phi*phi),           -(1-phi*phi));
    set_Qmat(Q22, 0.0, 2*(1-phi*phi)*(1-3*phi*phi), 2*phi*(1-phi*phi));

    // Set elements 1 and t of the vectors q and q2
    set_qvec(q,   1-phi,        (1-phi)*(1-phi));
    set_qvec(q2,  -(1-phi*phi), -2*(1-phi*phi)*(1-phi));

    for (t=0; t<n; t++) {

        // Form E1, b and delta polynomials as polynomials in (x_{t+1} - x_{t+1}^\circ)
        // giving values in terms of x_t
        poly_set(E1,    (mu0[t] - x0[t]), mud[t],           0.5*mudd[t]);
        poly_set(b,     (b[t] - x0[t]),   bd[t],            0.5*bdd[t]);
        poly_set(Sigma, Sigma[t],         Sigma[t] * sd[t], Sigma * 0.5*sdd[t]);


        poly_change_var(E1, );
        poly_change_var(b, )

        poly_sub(delta, E1, b);

        // Create intermediate polynomial products
        poly_mult(deltaSigma, delta, Sigma);
        poly_mult(bdeltaSigam, b, deltaSigma);
        poly_mult(delta2, delta, delta);
        poly_mult(Sigma2, Sigma, Sigma);

        // Computation of 

        // Computation of C11 = Cov[e_t, e_t^2|e_{t+1}]
        poly_set_scalar_mult(C11, 2.0, bV1);
        poly_add_scalar_mult(C11, 4.0, deltaSigma);

        // Computation of V2 = Var[e_t^2|e_{t+1}]
        poly_set_scalar_mult(V2, 4.0, b2V1);
        poly_add_scalar_mult(V2, 8.0, bdeltaSigma);
        poly_add_scalar_mult(V2, 2.0, Sigma2)
    }        
}
