#include <math.h>
#include "mex.h"

// The following functions are utilities for operations on 2nd order polynomials.
// They give 2nd order polynomial results; any higher order terms are dropped.
// The vector (p[0], p[1], p[2]) represents the polynomial
//
//    p[0] + p[1] x + p[2] x^2.

// polynomial assignment by element: p = (p0, p1, p2)
static inline void p_set(double *p, double p0, double p1, double p2)
{
    p[0] = p0; p[1] = p1; p[2] = p2;
}

// polynomial assignment: p = p1
static inline void p_copy(double *p, const double *p1)
{
    int i;
    for (i=0; i<3; i++)
        p[i] = p1[i];
}

// polynomial change of variables: take a polynomial in e+c and change in place
// to a polynomial in e, so that the result is the same function
static inline void p_change_var(double *p, double c)
{
    double p2c = p[2] * c;
    p[0] += (p2c + p[1]) * c;
    p[1] += 2*p2c;
}

// polynomial assignment with scalar multiplication: p = c p1
static inline void p_set_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<3; i++)
    p[i] = c * p1[i];
}

// polynomial addition with scalar multiplication p += c p1
static inline void p_add_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<3; i++)
    p[i] += c * p1[i];
}

// polynomial addition: p = p1 + p2
static inline void p_add(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<3; i++)
        p[i] = p1[i] + p2[i];
}

// polynomial subtraction: p = p1 - p2
static inline void p_subtract(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<3; i++)
        p[i] = p1[i] - p2[i];
}

// polynomial multiplication: p = p1 * p2
// note that 3rd and 4th order terms in result are dropped
static inline void p_mult(double *p, const double *p1, const double *p2)
{
    p[0] = p1[0]*p2[0];
    p[1] = p1[0]*p2[1] + p1[1]*p2[0];
    p[2] = p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0];
}

// polynomial square: p = p1 * p1 (slightly more efficient than using p_mult)
// again, 3rd and 4th order terms in result are dropped
static inline void p_square(double *p, const double *p1)
{
    p[0] = p1[0]*p1[0];
    p[1] = 2*p1[0]*p1[2];
    p[2] = p1[1]*p1[1] + 2*p1[0]*p[2];
}

// polynomial expectation operator
static inline void p_expect(
    // Output, a polynomial in e_t approximating E[p(e_{t-1}) | e_t]
    double *Ep,
    // Input, a polynomial in e_{t-1}
    double *p,
    // E[e_t|e_{t+1}] and E[e_t^2|e_{t+1}], as polynomials in e_t
    double E1, double E2
    )
{
    p_set_scalar_mult(Ep, p[2], E2):
    p_add_scalar_mult(Ep, p[1], E1);
    Ep[0] += p[0];
}

// polynomial covariance operator
static inline void p_cov(
    // Output, a polynomial in e_t approximating Cov[p1(e_{t-1}, p2(e_{t-1}) | e_t]
    double *Cp1p2,
    // Inputs, polynomials in e_{t-1}
    const double *p1, const double *p2,
    // Var[e_{t-1} | e_t], Cov[e_{t-1}, e_{t-1}^2 | e_t] and Var[e_{t-1}^2 | e_t]
    // as polynomials in e_t
    double *V1, double *C12, double *V2 
)
{
    p_add_scalar_mult(Cp1p2, p1[1] * p2[1], V1);
    p_add_scalar_mult(Cp1p2, p1[2] * p2[2], V2);
    p_add_scalar_mult(Cp1p2, p1[1] * p2[2] + p1[2] * p2[1], C12);
}

// (i, j) coordinates for each of six required covariances
// covariance fields c_tm1 and c_t set to zero
Cterm C[6] = {
    {0, 0, {0.0}, {0.0}},
    {0, 1, {0.0}, {0.0}},
    {1, 1, {0.0}, {0.0}},
    {0, 3, {0.0}, {0.0}},
    {1, 3, {0.0}, {0.0}},
    {3, 3, {0.0}, {0.0}}
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
    int t;
    double *b = mxStateGetPr(mxState,"b");
    double *bd = mxStateGetPr(mxState,"bd");
    double *bdd = mxStateGetPr(mxState,"bdd");
    double *mu0 = mxStateGetPr(mxState,"mu");
    double *mud = mxStateGetPr(mxState,"mud");
    double *mudd = mxStateGetPr(mxState,"mudd");
    double *Sigma = mxStateGetPr(mxState,"Sigma");
    double *sd = mxStateGetPr(mxState,"sd");
    double *sdd = mxStateGetPr(mxState,"sdd");

    double b[3], delta[3], deltaS[3], delta2[3], S2[3], bdeltaS[3];
    double E1[3], E2[3], C11[3], V1[3], V2[3];
    double bV1[3], b2V1[3];
    Qterm Q[5] = {0};

    // Initialization: store non-redundant elements of constant matrices Q, Q_2, and Q_{22}
    // and vectors q and q_2.
    // ------------------------------------------------------------------------------------

    // Set elements (1, 1), (t, t), (t, t+1) of the matrices Q, Q_2 and Q_{22}
    Q[0].Q_11 = 1.0;  Q[0].Q_tt = 1+phi*phi;                    Q[0].Q_ttp = -phi;
    Q[1].Q_11 = 0.0;  Q[1].Q_tt = 2*phi*(1-phi*phi);            Q[1].Q_tpp = -(1-phi*phi);
    Q[2].Q_11 = 0.0;  Q[2].Q_tt = 2*(1-phi*phi)*(1-3*phi*phi);  Q[2].Q_ttp = 2*phi*(1-phi*phi);

    // Set elements 1 and t of the vectors q and q_2
    Q[3].q_1 = 1-phi;         Q[3].q_t = (1-phi)*(1-phi);
    Q[4].q_1 = -(1-phi*phi);  Q[4].q_t = -2*(1-phi*phi)*(1-phi);

    for (t=0; t<n-1; t++) {
        int tp1 = t+1;

        // Part 1: compute polynomials approximating conditional moments of
        //   e_t given e_{t+1}
        // ----------------------------------------------------------------

        // Form E1, b and S polynomials as polynomials in
        //   (x_{t+1} - x_{t+1}^\circ)
        // E1 and b give conditional mean of x_t given x_{t+1}
        p_set(E1,  mu0[t] - mu[t],  mud[t],          0.5*mudd[t]);
        p_set(b,   b[t] - mu[t],    bd[t],           0.5*bdd[t]);
        p_set(S,   Sigma[tp1],      Sigma[t]*sd[t],  0.5*Sigma[t]*sdd[t]);

        // Convert E1, b and Sigma to polynomials in
        //    e_{t+1} \equiv (x_{t+1} - x_{t+1}^\circ) + (x_{t+1}^\circ - mu_t)
        //    giving values in terms of x_t, then 
        double c = x0[tp1] - mu[tp1];
        p_change_var(E1, c);
        p_change_var(b,  c);
        p_change_var(S,  c);

        // Compute polynomial delta, difference between conditional mean and mode
        p_subtract(delta, E1, b);

        // Create intermediate polynomial products
        p_mult(deltaS, delta, S);
        p_mult(bdeltaS, b, deltaS);
        p_square(E12, E1);
        p_square(delta2, delta);
        p_square(S2, S);

        // Computation of V1 = Var[e_t|e_{t+1}] and E2 = E[e_t^2|e_{t+1}]
        p_subtract(V1, S, delta2);
        p_add(E2, V1, E12);

        // Computation of C11 = Cov[e_t, e_t^2|e_{t+1}]
        p_mult(bV1, b, V1);
        p_set_scalar_mult(C11, 2.0, bV1);
        p_add_scalar_mult(C11, 4.0, deltaS);

        // Computation of V2 = Var[e_t^2|e_{t+1}]
        p_mult(b2V1, b, bV1);
        p_set_scalar_mult(V2, 4.0, b2V1);
        p_add_scalar_mult(V2, 8.0, bdeltaS);
        p_add_scalar_mult(V2, 2.0, S2);

        // Part 2: compute m_t^{(i)}(e_{t+1}) and c_t^{(i,j)}(e_{t+1}) polynomials
        // in sequential procedure
        // -----------------------------------------------------------------------

        // Compute m_t^{(i)}(e_{t+1}) for quadratic forms
        for (iQ = 0; iQ < 5; iQ++) {
            Q_term *Qi = &(Q[iQ]);

            // Add terms for e_t and e_t^2 in z(e_t, e_{t+1}) to \tilde{m}_{t-1}
            if (iQ < 3)
                Qi->m_tm1[2] += (t==0) Qi->Q_11 : Qi->Q_tt;
            else
                Qi->m_tm1[1] += (t==0) Qi->q_1 : Qi->q_t;

            // Compute \tilde{m}_t
            p_expect(Qi->m_t, Qi->m_tm1, E1, E2);

            // Add term for e_t e_{t+1} product for tridiagonal cases
            if (iQ < 3) {
                Qi->m_t[1] += Qi->Q_ttp * E1[0];
                Qi->m_t[2] += Qi->Q_ttp * E1[1];
            }
        }

        // Compute c_t^{(i,j)}(e_{t+1}) polynomials
        for (iC = 0; iC < 6; iC++) {
            C_term *Ci = &(C[iC]);
            Q_term *Qi = &(Q[C->i]), *Qj = &(Q[C->j]);
            p_expect(Ci->c_t, Ci->c_tm1, E1, E2);
            p_cov(Ci->c_t, Qi->m_tm1, Qj->m_tm1, V1, C12, V2);
            if (iC < 5) { // Qi is a tridiagonal quadratic form
                double V1_e = Qi->Q_ttp * Qj->m_tm1[1];
                double C12_e = Qi->Q_ttp * Qj->m_tm1[2];
                if (iC < 3) { // Qj is a tridiagonal quadratic form
                    double V1_e2 = Qi->Q_ttp * Qj->Q_ttp;
                    V1_e += Qi->m_tm1[1] * Qj->Q_ttp;
                    C12_e += Qi->m_tm1[2] * Qj->Q_ttp;
                    Ci->c_t[2] += V1_e2 * V1[0];
                }
                Ci->c_t[1] += V1_e * V1[0] + C12_e * C12[0];
                Ci->c_t[2] += V1_e * V1[1] + C12_e * C12[1];
            }
            p_copy(Ci->c _tm1, Ci->c_t);
        }
        for (iQ = 0; iQ < 5; iQ++)
            p_copy(Q[iQ].m_tm1, Q[iQ].m_t)
    } // for(t=0; t<n-1; t++)
    // Iteration n 

} // void compute_grad_Hess(...)
