#include <math.h>
#include <string.h>
#include "mex.h"
#include "new_grad_hess.h"

// The following functions are utilities for operations on 2nd order polynomials.
// They give 2nd order polynomial results; any higher order terms are dropped.
// The vector (p[0], p[1], p[2]) represents the polynomial
//
//    p[0] + p[1] x + p[2] x^2.

// polynomial assignment by element: p = (p0, p1, p2)
static inline void p_set(double *p, double p0, double p1, double p2, double p3, double p4)
{
    p[0] = p0; p[1] = p1; p[2] = p2; p[3] = p3, p[4] = p4;
}

// polynomial assignment with scalar multiplication: p = c p1
static inline void p_set_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<5; i++)
    p[i] = c * p1[i];
}

// polynomial addition with scalar multiplication p += c p1
static inline void p_add_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<5; i++)
    p[i] += c * p1[i];
}

// polynomial addition: p = p1 + p2
static inline void p_add(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<5; i++)
        p[i] = p1[i] + p2[i];
}

// polynomial subtraction: p = p1 - p2
static inline void p_subtract(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<5; i++)
        p[i] = p1[i] - p2[i];
}

// polynomial multiplication: p = p1 * p2
// note that 3rd and 4th order terms in result are dropped
static inline void p_mult(double *p, const double *p1, const double *p2)
{
    p[0] = p1[0]*p2[0];
    p[1] = p1[0]*p2[1] + p1[1]*p2[0];
    p[2] = p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0];
    p[3] = 0.0;
    p[4] = 0.0;
}

static inline void p_mult3(double *p, const double *p1, const double *p2)
{
    p[0] = p1[0]*p2[0];
    p[1] = p1[0]*p2[1] + p1[1]*p2[0];
    p[2] = p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0];
    p[3] = p1[0]*p2[3] + p1[1]*p2[2] + p1[2]*p2[1] + p1[3]*p2[0];
    p[4] = 0.0;
}

// polynomial square: p = p1 * p1 (slightly more efficient than using p_mult)
// again, 3rd and 4th order terms in result are dropped
static inline void p_square(double *p, const double *p1)
{
    p[0] = p1[0]*p1[0];
    p[1] = 2*p1[0]*p1[1];
    p[2] = p1[1]*p1[1] + 2*p1[0]*p1[2];
    p[3] = 2*p1[0]*p1[3] + 2*p1[1]*p1[2];
    p[4] = 0.0;
}

// polynomial expectation operator
static inline void p_expect(
    // Output, a polynomial in e_{t+1} approximating E[p(e_t) | e_{t+1}]
    double *Ep,
    // Input, a polynomial in e_t
    double *p,
    // E[e_t|e_{t+1}] and E[e_t^2|e_{t+1}], as polynomials in e_{t+1}
    const double *E1, const double *E2, const double *E3
    )
{
    p_set_scalar_mult(Ep, p[2], E2);
    p_add_scalar_mult(Ep, p[1], E1);
    p_add_scalar_mult(Ep, p[3], E3);
    Ep[0] += p[0];
}

// polynomial covariance operator
static inline void p_cov(
    // Output, a polynomial in e_{t+1} approximating Cov[p1(e_t, p2(e_t) | e_{t+1}]
    double *Cp1p2,
    // Inputs, polynomials in e_t
    const double *p1, const double *p2,
    // Var[e_t | e_{t+1}], Cov[e_t, e_t^2 | e_{t+1}] and Var[e_t^2 | e_{t+1}]
    // as polynomials in e_{t+1}
    const double *V1, const double *C12, const double *C13, const double *V2 
)
{
    p_add_scalar_mult(Cp1p2, p1[1] * p2[1], V1);
    p_add_scalar_mult(Cp1p2, p1[2] * p2[2], V2);
    p_add_scalar_mult(Cp1p2, p1[1] * p2[2] + p1[2] * p2[1], C12);
    p_add_scalar_mult(Cp1p2, p1[1] * p2[3] + p1[3] * p2[1], C13);
}

// print a polynomial (in a variable e) to the Matlab console
static inline void poly_print(char *s, double *poly)
{
    mexPrintf("%s: %lf + %lf e + %lf e^2 + %lf e^3 + %lf e^4\n", s,
        poly[0], poly[1], poly[2], poly[3], poly[4]);
}

static inline void Q_print(char *s, Q_term *Q)
{
    mexPrintf("%s: Q_11 = %lf, Q_tt = %lf, Q_ttp = %lf, q_1 = %lf, q_t = %lf\n",
        s, Q->Q_11, Q->Q_tt, Q->Q_ttp, Q->q_1, Q->q_t);
    mexPrintf("\tm_t = (%lf, %lf, %lf, %lf)\n", Q->m_t[0], Q->m_t[1], Q->m_t[2], Q->m_t[3]);
    mexPrintf("\tm_tm1 = (%lf, %lf, %lf, %lf)\n\n", Q->m_tm1[0], Q->m_tm1[1], Q->m_tm1[2], Q->m_tm1[3]);
}

static inline double *mxStateGetPr(const mxArray *mxState, char *field_name)
{
    mxArray *field_pr = mxGetField(mxState, 0, field_name);
    return mxGetPr(field_pr);
}

void compute_new_grad_Hess(
    const mxArray *mxState,
    int n,
    double *mu,   // Prior mean of x, as a vector
    double phi,   // Autocorrelation parameter of x_t of x_t process
    double omega, // Innovation precision parameter of x_t process
    // Output
    double *grad, // 3 x 1 vector, approximation of E[g_{x|\theta}(\theta)]
    double *Hess, // 3 x 3 matrix, approximation of E[H_{x|\theta)(\theta)]
    double *var   // 3 x 3 matrix, approximation of Var[g_{x|\theta}(\theta)]
    )
{
    int t, iQ, iC;
    double *x0 = mxStateGetPr(mxState,"x_mode");
    double *ad = mxStateGetPr(mxState,"ad");
    double *b0 = mxStateGetPr(mxState,"b");     // Value,
    double *bd = mxStateGetPr(mxState,"bd");   // 1st derivative,
    double *bdd = mxStateGetPr(mxState,"bdd"); // and 2nd derivative, conditional mode
    double *bddd = mxStateGetPr(mxState,"bddd"); // and 2nd derivative, conditional mode
    double *mu0 = mxStateGetPr(mxState,"mu");  // Same for conditional mean
    double *mud = mxStateGetPr(mxState,"mud");
    double *mudd = mxStateGetPr(mxState,"mudd");
    double *Sigma = mxStateGetPr(mxState,"Sigma");
    double *sd = mxStateGetPr(mxState,"sd");   // 1st derivative of log(Sigma)
    double *sdd = mxStateGetPr(mxState,"sdd"); // 2nd derivative of log(Sigma)
    double *sddd = mxStateGetPr(mxState,"sddd"); // 2nd derivative of log(Sigma)

    // Polynomials for conditional moments of e_t given e_{t+1}
    double E1[5], E2[5], E3[5], C12[5], C13[5], V1[5], V2[5];
    double b[5], delta[5], S[5], delta_S[5], delta2[5], S2[5], b_delta_S[5];
    double E12[5], E1_E12[5], E1_E3[5], b_V1[5], b2_V1[5];

    // Initialization: store non-redundant elements of constant matrices Q, Q_2, and Q_{22}
    // and vectors q and q_2.
    // ------------------------------------------------------------------------------------
    Q_term Q[5] = {0}; // Information about Q, Q_2, Q_{22}, q, q_2

    // Set elements (1, 1), (t, t), (t, t+1) of the matrices Q, Q_2 and Q_{22}
    Q[0].Q_11 = 1.0;  Q[0].Q_tt = 1+phi*phi;                    Q[0].Q_ttp = -2*phi;
    Q[1].Q_11 = 0.0;  Q[1].Q_tt = 2*phi*(1-phi*phi);            Q[1].Q_ttp = -2*(1-phi*phi);
    Q[2].Q_11 = 0.0;  Q[2].Q_tt = 2*(1-phi*phi)*(1-3*phi*phi);  Q[2].Q_ttp = 4*phi*(1-phi*phi);

    // Set elements 1 and t of the vectors q and q_2
    Q[3].q_1 = 1-phi;         Q[3].q_t = (1-phi)*(1-phi);
    Q[4].q_1 = -(1-phi*phi);  Q[4].q_t = -2*(1-phi*phi)*(1-phi);

    // (i, j) coordinates for each of six required covariances
    // The covariance fields c_tm1 and c_t are set to zero
    C_term C[6] = {
        {0, 0, {0.0}, {0.0}},  // 0, for Var[e^\top Q e]
        {0, 1, {0.0}, {0.0}},  // 1, for Cov[e^\top Q e, e^\top Q_2 e]
        {1, 1, {0.0}, {0.0}},  // 2, for Var[e^\top Q_2 e]
        {0, 3, {0.0}, {0.0}},  // 3, for Cov[e^\top Q e, q e]
        {1, 3, {0.0}, {0.0}},  // 4, for Cov[e^\top Q_2 e, q e]
        {3, 3, {0.0}, {0.0}}   // 5  for Var[q e]
    };

    // Variables for computing d statistics
    double d1 = x0[0] - mu[0], dn = x0[n-1] - mu[n-1];
    double dt = d1, dtm1 = 0.0, dtp1 = 0.0, dt_sum = 0.0, dt2_sum = 0.0, dttp_sum = 0.0;

    for (t=0; t<n; t++) {

        // Part 1: compute polynomials E1, E2, V1, V2, C12 approximating
        // E[e_t|e_{t+1], E[e_t^2|e_{t+1}], Var[e_t|e_{t+1}], Var[e_t|e_{t+1}]
        // and Cov[e_t,e_t^2|e_{t+1}]. Polynomials are 2nd order in e_{t+1}
        // -------------------------------------------------------------------

        // Form E1, b and S polynomials as polynomials in
        //   (x_{t+1} - x_{t+1}^\circ)
        // E1 and b give conditional mean and mode of x_t given x_{t+1}
        double S0 = Sigma[t], S20;
        double bd_ad = bd[t]/ad[t];
        if (t<n-1) {
            dtp1 = x0[t+1] - mu[t+1];
            dttp_sum += dt * dtp1;
            dt_sum += dt;
            dt2_sum += dt * dt;
            p_set(E1,  mu0[t] - x0[t],  mud[t],         0.5*mudd[t],  bddd[t]/6.0, 0.0);
            p_set(b,   b0[t] - x0[t],   bd[t],          0.5*bdd[t],   bddd[t]/6.0, 0.0);

            //S0 *= exp(sd[t]*b[0]/ad[t]);
            S20 = S0*S0;
            p_set(S,   S0,              S0*sd[t],        0.5*S0*sdd[t], S0*sddd[t]/6.0, 0.0);
            p_set(S2,  S20,             2*S20*sd[t],     S20*sdd[t],    S20*sddd[t]/3.0, 0.0);
        }
        else { // Last value is unconditional.
            dtp1 = 0.0;
            p_set(E1,  mu0[t] - x0[t],  0.0,             0.0, 0.0, 0.0);
            p_set(b,   b0[t] - x0[t],   0.0,             0.0, 0.0, 0.0);
            p_set(S,   S0,              0.0,             0.0, 0.0, 0.0);
            p_set(S2,  S20,             0.0,             0.0, 0.0, 0.0);
        }

        // Compute polynomial delta, difference between conditional mean and mode
        p_subtract(delta, E1, b);

        // Compute intermediate polynomial products
        p_mult3(delta_S, delta, S);
        p_mult3(b_delta_S, b, delta_S);
        p_square(E12, E1);
        p_square(delta2, delta);

        // Compute V1 = Var[e_t|e_{t+1}] and E2 = E[e_t^2|e_{t+1}]
        p_subtract(V1, S, delta2);
        p_add(E2, V1, E12);

        // Computation of C12 = Cov[e_t, e_t^2|e_{t+1}]
        p_mult3(b_V1, b, V1);
        p_set_scalar_mult(C12, 2.0, b_V1);
        p_add_scalar_mult(C12, 4.0, delta_S);

        p_mult3(E1_E12, E1, E12);
        p_set_scalar_mult(E3, 1.0, C12);
        p_add_scalar_mult(E3, 1.0, E1_E12);

        // Computation of V2 = Var[e_t^2|e_{t+1}]
        p_mult3(b2_V1, b, b_V1);
        p_set_scalar_mult(V2, 4.0, b2_V1);
        p_add_scalar_mult(V2, 8.0, b_delta_S);
        p_add_scalar_mult(V2, 2.0, S2);

        // Computation of C13 = Cov[e_t, e_t^3|e_{t+1}]
        p_square(C13, E2);
        p_add_scalar_mult(C13, 1.0, V2);
        p_mult3(E1_E3, E1, E3);
        p_add_scalar_mult(C13, -1.0, E1_E3);
        p_set(C13, 0.0, 0.0, 0.0, 0.0, 0.0);

        if (t%1000 == 0) {
            poly_print("E1", E1);
            poly_print("b", b);
            poly_print("S", S);
            poly_print("V1", V1);
            poly_print("E2", E2);
            poly_print("E3", E3);
            poly_print("V2", V2);
            poly_print("C12", C12);
            poly_print("C13", C13);
        }

        // Part 2: compute m_t^{(i)}(e_{t+1}) and c_t^{(i,j)}(e_{t+1}) polynomials
        // in sequential procedure
        // -----------------------------------------------------------------------

        // Compute m_t^{(i)}(e_{t+1}) for quadratic forms
        for (iQ = 0; iQ < 5; iQ++) {
            Q_term *Qi = &(Q[iQ]);

            // Add terms for e_t and e_t^2 in z^{(i)}(e_t, e_{t+1}) to
            // \tilde{m}^{(i)}_{t-1} and compute all terms of \tilde{m}^{(i)}_t
            // except the one with the expectation of e_t e_{t+1} 
            if (iQ < 3) {
                double Q_tt = ((t==0) || (t==n-1)) ? Qi->Q_11 : Qi->Q_tt;
                Qi->m_tm1[1] += 2*Q_tt * dt + Qi->Q_ttp * (dtm1 + dtp1);
                Qi->m_tm1[2] += Q_tt;
            }
            else
                Qi->m_tm1[1] += ((t==0) || (t==n-1)) ? Qi->q_1 : Qi->q_t;
            p_expect(Qi->m_t, Qi->m_tm1, E1, E2, E3);

            // Add term for e_t e_{t+1} product for tridiagonal cases
            if (iQ < 3 && t<n-1) {
                Qi->m_t[1] += Qi->Q_ttp * E1[0];
                Qi->m_t[2] += Qi->Q_ttp * E1[1];
                Qi->m_t[3] += Qi->Q_ttp * E1[2];
            }
        }

        // Compute c_t^{(i,j)}(e_{t+1}) polynomials
        for (iC = 0; iC < 6; iC++) {
            C_term *Ci = &(C[iC]);
            Q_term *Qi = &(Q[Ci->i]), *Qj = &(Q[Ci->j]);

            // c_t^{(i,j)} = E[c_{t-1}^{i,j} | e_{t+1}] ...
            p_expect(Ci->c_t, Ci->c_tm1, E1, E2, E3);
            // ... + Cov[m_{t-1}^{(i)} + <z_i>, m_{t-1}^{(j)} + <z_j> | e_{t+1}],
            // where <z_i> is z_t^{(i)} less the e_t e_{t+1} term, same for j
            p_cov(Ci->c_t, Qi->m_tm1, Qj->m_tm1, V1, C12, C13, V2);

            // Add term for e_t e_{t+1} product in tridiagonal cases
            if (iC < 5 && t < n-1) { // Qi is a tridiagonal quadratic form
                double V1_e = Qi->Q_ttp * Qj->m_tm1[1];    // Coeff of e_{t+1} var[e_t]
                double C12_e = Qi->Q_ttp * Qj->m_tm1[2];   // Coeff of e_{t+1} cov[e_t, e_t^2]
                double C13_e = Qi->Q_ttp * Qj->m_tm1[3];   // Coeff of e_{t+1} cov[e_t, e_t^3]
                if (iC < 3) { // Qj is a tridiagonal quadratic form
                    double V1_e2 = Qi->Q_ttp * Qj->Q_ttp;  // Coeff of e_{t+1} var[e_t^2]
                    V1_e += Qi->m_tm1[1] * Qj->Q_ttp;
                    C12_e += Qi->m_tm1[2] * Qj->Q_ttp;
                    C13_e += Qi->m_tm1[3] * Qj->Q_ttp;
                    Ci->c_t[2] += V1_e2 * V1[0];
                    Ci->c_t[3] += V1_e2 * V1[1];
                }
                Ci->c_t[1] += V1_e * V1[0] + C12_e * C12[0] + C13_e * C13[0];
                Ci->c_t[2] += V1_e * V1[1] + C12_e * C12[1] + C13_e * C13[1];
                Ci->c_t[3] += V1_e * V1[2] + C12_e * C12[2] + C13_e * C13[2];
            }
            memcpy(Ci->c_tm1, Ci->c_t, 4 * sizeof(double));
        }
        dtm1 = dt;
        dt = dtp1;
        for (iQ = 0; iQ < 5; iQ++)
            memcpy(Q[iQ].m_tm1, Q[iQ].m_t, 4 * sizeof(double));
    } // for(t=0; t<n-1; t++)

    // To avoid double counting
    dt_sum -= d1;
    dt2_sum -= d1 * d1;
    for (iQ=0; iQ<3; iQ++) {
        // Compute constant part of quadratic forms
        Q[iQ].dQd = Q[iQ].Q_11 * (d1*d1 + dn*dn);
        Q[iQ].dQd += Q[iQ].Q_tt * dt2_sum + Q[iQ].Q_ttp * dttp_sum;
    }
    // ... and linear forms.
    for (iQ=3; iQ<5; iQ++) {
        Q[iQ].qd = Q[iQ].q_1 * (d1 + dn) + Q[iQ].q_t * dt_sum;
    }

    // Flat indices for a 3 x 3 matrix
    // 0 1 2
    // 3 4 5
    // 6 7 8

    // Assign elements of expected gradient
    grad[0] = 0.5*n - 0.5 * omega * (Q[0].dQd + Q[0].m_t[0]); 
    grad[1] = -phi - 0.5 * omega * (Q[1].dQd + Q[1].m_t[0]);
    grad[2] = omega * (Q[3].qd + Q[3].m_t[0]);

    mexPrintf("Q[2].dQd: %lf, Q[2].m_t[0]: %lf\n", Q[2].dQd, Q[2].m_t[0]);

    // Assign elements of expected Hessian
    Hess[0] = -0.5 * omega * (Q[0].dQd + Q[0].m_t[0]);
    Hess[1] = Hess[3] = -0.5 * omega * (Q[1].dQd + Q[1].m_t[0]);
    Hess[4] = -(1-phi*phi) - 0.5 * omega * (Q[2].dQd + Q[2].m_t[0]);
    Hess[2] = Hess[6] = omega * (Q[3].qd + Q[3].m_t[0]);
    Hess[5] = Hess[7] = omega * (Q[4].qd + Q[4].m_t[0]);
    Hess[8] = -omega * (1-phi) * (n * (1-phi) + 2*phi);

    // Assign elements of variance of gradient
    var[0] = 0.25 * omega * omega * C[0].c_t[0];
    var[1] = var[3] = 0.25 * omega * omega * C[1].c_t[0];
    var[4] = 0.25 * omega * omega * C[2].c_t[0];
    var[2] = var[6] = 0.5 * omega * omega * C[3].c_t[0];
    var[5] = var[7]= 0.5 * omega * omega * C[4].c_t[0];
    var[8] = omega * omega * C[5].c_t[0];
} // void compute_grad_Hess(...)
