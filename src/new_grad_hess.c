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

// polynomial change of variables: take a polynomial in e+c and change in place
// to a polynomial in e, so that the result is the same function
static inline void poly_change_var(double *poly, double c)
{
    double p2c = poly[2] * c;
    poly[0] += (p2c + poly[1]) * c;
    poly[1] += 2*p2c;
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

// square of polynomial
static inline void poly_square(double *poly, const double *poly1)
{
    poly[0] = poly1[0]*poly1[0];
    poly[1] = 2*poly1[0]*poly1[2];
    poly[2] = poly1[1]*poly1[1] + 2*poly1[0]*poly[2];
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
    double b[3], delta[3], deltaS[3], delta2[3], S2[3], bdeltaS[3];
    double E1[3], E2[3], C11[3], V1[3], V2[3];
    double bV1[3], b2V1[3];
    Qterm Q[5] = {0};

    // Initialization, store non-redundant elements of constant matrices Q, Q_2, and Q_{22}
    // and vectors q and q_2.
    // ------------------------------------------------------------------------------------

    // Set elements (1, 1), (t, t), (t, t+1) of the matrices Q, Q_2 and Q_{22}
    Q[0].Q_11 = 1.0;  Q[0].Q_tt = 1+phi*phi;Q[0].Q_ttp = -phi;
    Q[1].Q_11 = 0.0;  Q[1].Q_tt = 2*phi*(1-phi*phi);            Q[1].Q_tpp = -(1-phi*phi);
    Q[2].Q_11 = 0.0;  Q[2].Q_tt = 2*(1-phi*phi)*(1-3*phi*phi);  Q[2].Q_ttp = 2*phi*(1-phi*phi));

    // Set elements 1 and t of the vectors q and q_2
    Q[3].q_1 = 1-phi;         Q[3].q_t = (1-phi)*(1-phi);
    Q[4].q_1 = -(1-phi*phi);  Q[4].q_t = -2*(1-phi*phi)*(1-phi);

    for (t=0; t<n-1; t++) {
        int tp1 = t+1;

        // Part 1, compute polynomials approximating conditional moments of
        //   e_t given e_{t+1}
        // ----------------------------------------------------------------

        // Form E1, b and S polynomials as polynomials in
        //   (x_{t+1} - x_{t+1}^\circ)
        // E1 and b give conditional mean of x_t given x_{t+1}
        poly_set(E1,  mu0[t] - mu[t],  mud[t],          0.5*mudd[t]);
        poly_set(b,   b[t] - mu[t],    bd[t],           0.5*bdd[t]);
        poly_set(S,   Sigma[tp1],      Sigma[t]*sd[t],  0.5*Sigma[t]*sdd[t]);

        // Convert E1, b and Sigma to polynomials in
        //    e_{t+1} \equiv (x_{t+1} - x_{t+1}^\circ) + (x_{t+1}^\circ - mu_t)
        //    giving values in terms of x_t, then 
        double c = x0[tp1] - mu[tp1];
        poly_change_var(E1, c);
        poly_change_var(b,  c);
        poly_change_var(S,  c);

        // Compute polynomial delta, difference between conditional mean and mode
        poly_sub(delta, E1, b);

        // Create intermediate polynomial products
        poly_mult(deltaS, delta, S);
        poly_mult(bdeltaS, b, deltaS);
        poly_square(E12, E1);
        poly_square(delta2, delta);
        poly_square(S2, S);

        // Computation of V1 = Var[e_t|e_{t+1}] and E2 = E[e_t^2|e_{t+1}]
        poly_sub(V1, S, delta2);
        poly_add(E2, V1, E12);
        poly_mult(bV1, b, V1);
        poly_mult(b2V1, b, bV1);

        // Computation of C11 = Cov[e_t, e_t^2|e_{t+1}]
        poly_set_scalar_mult(C11, 2.0, bV1);
        poly_add_scalar_mult(C11, 4.0, deltaS);

        // Computation of V2 = Var[e_t^2|e_{t+1}]
        poly_set_scalar_mult(V2, 4.0, b2V1);
        poly_add_scalar_mult(V2, 8.0, bdeltaS);
        poly_add_scalar_mult(V2, 2.0, S2);

        // Part 2, compute m_t^{(i)}(e_{t+1}) and c_t^{(i,j)}(e_{t+1}) polynomials
        // in sequential procedure
        // -----------------------------------------------------------------------

        // Compute m_t^{(i)}(e_{t+1}) for quadratic forms
        for (pQ = Q, iQ = 0; iQ < 5; pQ++, iQ++) {

            // Add terms for e_t and e_t^2 in z(e_t, e_{t+1}) to \tilde{m}_{t-1}
            if (iQ < 3)
                pQ->m_tm1[2] += (t==0) pQ->Q_11 : pQ->Q_tt;
            else
                pQ->m_tm1[1] += (t==0) pQ->q_1 : pQ->q_t;

            // Compute \tilde{m}_t
            poly_set_scalar_mult(pQ->m_t, pQ->m_tm1[2], E2):
            poly_add_scalar_mult(pQ->m_t, pQ->m_tm1[1], E1);
            pQ->m_t[0] += pQ->m_tm1[0];
            
            // Add term for e_t e_{t+1} product
            if (iQ < 3) {
                Q[iQ].m_t[1] += Q[iQ].Q_ttp * E1[0];
                Q[iQ].m_t[2] += Q[iQ].Q_ttp * E1[1];
            }
        }

        // Compute c_t^{(i,j)}(e_{t+1}) polynomials
    }
}
