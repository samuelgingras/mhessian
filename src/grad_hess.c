#include <math.h>
#include <string.h>
#include "mex.h"
#include "state.h"
#include "grad_hess.h"

// The following functions are utilities for operations on 4th order polynomials.
// They give 4th order polynomial results; any higher order terms are dropped.
// The vector (p[0], p[1], p[2], p[3]) represents the polynomial
//
//    p[0] + p[1] x + p[2] x^2 + p[3] x^3.

// polynomial assignment by element: p = (p0, p1, p2, p3, p4s)
static inline void p_set(double *p, double p0, double p1, double p2, double p3)
{
    p[0] = p0; p[1] = p1; p[2] = p2; p[3] = p3;
}

// polynomial assignment with scalar multiplication: p = c p1
static inline void p_set_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<p_len; i++)
    p[i] = c * p1[i];
}

// polynomial addition with scalar multiplication p += c p1
static inline void p_add_scalar_mult(double *p, double c, const double *p1)
{
  int i;
  for (i=0; i<p_len; i++)
    p[i] += c * p1[i];
}

// polynomial addition: p = p1 + p2
static inline void p_add(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<p_len; i++)
        p[i] = p1[i] + p2[i];
}

// polynomial subtraction: p = p1 - p2
static inline void p_subtract(double *p, const double *p1, const double *p2)
{
    int i;
    for (i=0; i<p_len; i++)
        p[i] = p1[i] - p2[i];
}

// Polynomial multiplication followed by addition: p = p + p1 * p2
static inline void p_mult_add(double *p, const double *p1, const double *p2)
{
    p[0] += p1[0]*p2[0];
    p[1] += p1[0]*p2[1] + p1[1]*p2[0];
    p[2] += p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0];
    p[3] += p1[0]*p2[3] + p1[1]*p2[2] + p1[2]*p2[1] + p1[3]*p2[0];
}

// polynomial square: p = p1 * p1 (slightly more efficient than using p_mult)
// Again, higher orders than x^3 are dropped.
static inline void p_square(double *p, const double *p1)
{
    double p0_2 = 2.0 * p1[0];
    p[0] = p1[0]*p1[0];
    p[1] = p0_2*p1[1];
    p[2] = p1[1]*p1[1] + p0_2*p1[2];
    p[3] = p0_2*p1[3] + 2*p1[1]*p1[2];
}

// polynomial expectation operator
static inline void p_expect(
    // Output, a polynomial in e_{t+1} approximating E[p(e_t) | e_{t+1}]
    double *Ep,
    // Input, a polynomial in e_t
    double *p,
    // E[e_t|e_{t+1}], E[e_t^2|e_{t+1}], E[e_t^3|e_{t+1}] as polynomials in e_{t+1}
    const double *E1, const double *E2, const double *E3
    )
{
    p_set_scalar_mult(Ep, p[1], E1);
    p_add_scalar_mult(Ep, p[2], E2);
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
    const double *V1, const double *C12, const double *V2 
)
{
    p_add_scalar_mult(Cp1p2, p1[1] * p2[1], V1);
    p_add_scalar_mult(Cp1p2, p1[2] * p2[2], V2);
    p_add_scalar_mult(Cp1p2, p1[1] * p2[2] + p1[2] * p2[1], C12);
}

// print a polynomial (in a variable e) to the Matlab console
static void p_print(char *s, double *p)
{
    mexPrintf("%s: %lf + %lf e + %lf e^2 + %lf e^3\n", s, p[0], p[1], p[2], p[3]);
}

// print m_t and m_tm1 polynomials for a given quadratic or linear form
static void Q_print(char *s, Q_term *Q)
{
    mexPrintf("%s: Q_11 = %lf, Q_tt = %lf, Q_ttp = %lf, q_1 = %lf, q_t = %lf\n",
        s, Q->Q_11, Q->Q_tt, Q->Q_ttp, Q->q_1, Q->q_t);
    mexPrintf("\tm_t = (%lf, %lf, %lf, %lf)\n", Q->m_t[0], Q->m_t[1], Q->m_t[2], Q->m_t[3]);
    mexPrintf("\tm_tm1 = (%lf, %lf, %lf, %lf)\n\n", Q->m_tm1[0], Q->m_tm1[1], Q->m_tm1[2], Q->m_tm1[3]);
}


void compute_grad_Hess(
    // Input
    int long_th,
    // long_th = FALSE for theta = (ln(omega), atanh(phi))),
    // long_th = TRUE  for theta = (ln(omega), atanh(phi)), mu)
    State *state,   // Structure with derivatives
    Theta *theta,   // Structure with parameters
    // Output
    double *grad, // Vector, approximation of E[g_{x|\theta}(\theta)]
    double *Hess, // Matrix, approximation of E[H_{x|\theta)(\theta)]
    double *var,  // Matrix, approximation of Var[g_{x|\theta}(\theta)]
    // In following outputs d_t is defined as (x0_t - mu)
    double *d1n_sum,     // (x0_1-mu) + (x0_n-mu)
    double *dt_sum,      // (x0_2-mu) + ... + (x_{n-1}-mu)
    double *d11nn_sum,   // (x0_1-mu)^2 + (x0_n-mu)^2
    double *dtt_sum,     // (x0_2-mu)^2 + ... + (x0_{n-1}-mu)^2
    double *dttp_sum,    // (x0_1-mu)(x0_2-mu) + ... + (x0_{n-1}-mu)(x0_n-mu)
    // Output to implement cumulative gradient computations
    double *g_cum,
    // Output to implement third derivative stuff
    double *cov_Q1Q2
    )
{
    double T[27];                   // For third derivative information
    int t, iQ, iC;
    int n = state->n;               // Number of observations
    double *x0 = state->alC;        // Mode
    double *ad = state->ad;
    double *b0 = state->b;;         // Value,
    double *bd = state->bd;         // 1st derivative,
    double *bdd = state->bdd;       // 2nd derivative, conditional mode
    double *bddd = state->bddd;     // 3rd derivative, conditional mode
    double *mu0 = state->mu;        // Same for conditional mean
    double *mud = state->mud;       
    double *mudd = state->mudd;
    double *Sigma = state->Sigma;
    double *sd = state->sd;         // 1st derivative of log(Sigma)
    double *sdd = state->sdd;       // 2nd derivative of log(Sigma)
    double *sddd = state->sddd;     // 3rd derivative of log(Sigma)
             
    double *mu = theta->x->mu_tm;       // Prior mean of x, as a vector
    double phi = theta->x->phi;         // Autocorrelation parameter of x_t process
    double omega = theta->x->omega;     // Innovation precision parameter of x_t process

    int nQ = long_th ? 5 : 3;
    int nC = long_th ? 7 : 3; // Was 6 : 3. Need to put (1,2) as 4th, not 7th C

    // Polynomials for conditional moments of e_t given e_{t+1}
    double b[p_len] = {0}, delta[p_len] = {0}, S[p_len] = {0};   // Given
    double b_2[p_len], S2[p_len], E12[p_len];  // Intermediate polynomials
    double E1[p_len], E2[p_len], E3[p_len], C12[p_len], V1[p_len], V2[p_len]; // Direct moments

    // Initialization: store non-redundant elements of constant matrices Q, Q', and Q''
    // and vectors q and q_2.
    // ------------------------------------------------------------------------------------
    Q_term Q[5] = {0}; // Information about Q, Q', Q'', q, q'

    // Set elements (1, 1), (t, t), (t, t+1) of the matrices Q, Q' and Q''
    Q[0].Q_11 = 1.0;  Q[0].Q_tt = 1+phi*phi;                    Q[0].Q_ttp = -2*phi;
    Q[1].Q_11 = 0.0;  Q[1].Q_tt = 2*phi*(1-phi*phi);            Q[1].Q_ttp = -2*(1-phi*phi);
    Q[2].Q_11 = 0.0;  Q[2].Q_tt = 2*(1-phi*phi)*(1-3*phi*phi);  Q[2].Q_ttp = 4*phi*(1-phi*phi);

    // Set elements 1 and t of the vectors q and q_2
    Q[3].q_1 = 1-phi;         Q[3].q_t = (1-phi)*(1-phi);
    Q[4].q_1 = -(1-phi*phi);  Q[4].q_t = -2*(1-phi*phi)*(1-phi);

    // (i, j) coordinates for each of six required covariances
    // The covariance fields c_tm1 and c_t are set to zero
    C_term C[7] = {
        {0, 0, {0.0}, {0.0}},  // 0, for Var[e^\top Q e]
        {0, 1, {0.0}, {0.0}},  // 1, for Cov[e^\top Q e, e^\top Q' e]
        {1, 1, {0.0}, {0.0}},  // 2, for Var[e^\top Q' e]
        {0, 3, {0.0}, {0.0}},  // 3, for Cov[e^\top Q e, q e]
        {1, 3, {0.0}, {0.0}},  // 4, for Cov[e^\top Q' e, q e]
        {3, 3, {0.0}, {0.0}},  // 5, for Var[q e]
        {1, 2, {0.0}, {0.0}}   // 6, for Cov[e^\top Q'e, e^\top Q''e]
    };

    // Forward pass to get g_cum
    double x_bar = 0.0;
    for (t=0; t<n; t++)
        x_bar += x0[t];
    x_bar /= n;
    double Etp = (mu0[n-1] - x0[n-1]), Vtp = Sigma[n-1];
    double deltp = Etp + x0[n-1] - x_bar;
    g_cum[n-1] = (1-phi*phi) * Vtp;
    g_cum[2*n-1] = phi * Vtp;
    g_cum[3*n-1] = (1-phi*phi) * deltp * deltp;
    g_cum[4*n-1] = phi * deltp * deltp;
    g_cum[5*n-1] = Vtp;
    for (t=n-2; t>=0; t--) {
        double Et = (mu0[t] - x0[t]) + mud[t] * Etp;
        double EVttp = Sigma[t] * (1 + sd[t]*Etp);
        double Vt = EVttp + mud[t]*mud[t] * Vtp;
        double mud_phi = mud[t] - phi;
        g_cum[t] = EVttp + mud_phi*mud_phi * Vtp;
        g_cum[t+n] = mud_phi * Vtp;
        double delt = Et + x0[t] - x_bar;
        g_cum[t+2*n] = pow(delt - phi*deltp, 2);
        g_cum[t+3*n] = (delt - phi*deltp) * deltp;
        g_cum[t+4*n] = Vt;
        Etp = Et; Vtp = Vt; deltp = delt;
    }

    // Variables for computing d statistics
    double d1 = x0[0] - mu[0], dn = x0[n-1] - mu[n-1];
    double dt = d1, dtm1 = 0.0, dtp1 = 0.0;
    *dt_sum = 0.0;
    *dtt_sum = 0.0;
    *dttp_sum = 0.0;
    *d1n_sum = d1 + dn;
    *d11nn_sum = d1*d1 + dn*dn;

    for (t=0; t<n; t++) {

        // Part 1: compute polynomials E1, E2, V1, V2, C12 approximating
        // E[e_t|e_{t+1], E[e_t^2|e_{t+1}], Var[e_t|e_{t+1}], Var[e_t|e_{t+1}]
        // and Cov[e_t,e_t^2|e_{t+1}]. Polynomials are 2nd order in e_{t+1}
        // -------------------------------------------------------------------

        // Form E1, b and S polynomials as polynomials in
        //   (x_{t+1} - x_{t+1}^\circ)
        // E1 and b give conditional mean and mode of x_t given x_{t+1}
        double S0 = Sigma[t], S20;
        double dt_2 = 2.0 * dt;
        if (t<n-1) {
            dtp1 = x0[t+1] - mu[t+1];
            *dt_sum += dt;
            *dtt_sum += dt * dt;
            *dttp_sum += dt * dtp1;

            // Set E1, b
            E1[0] = mu0[t] - x0[t];    b[0] = b0[t] - x0[t];
            E1[1] = mud[t];            b[1] = bd[t];
            E1[2] = 0.5*mudd[t];       b[2] = 0.5*bdd[t];
            E1[3] =                    b[3] = (1.0/6.0) * bddd[t]/6.0;

            // Set S, S2
            S0 *= 1.0 + sd[t]*b[0]/ad[t];
            S20 = S0*S0;
            S[0] = S0;                    S2[0] = S20;
            S[1] = S0*sd[t];              S2[1] = 2*S20*sd[t];
            S[2] = 0.5*S0*sdd[t];         S2[2] = S20*sdd[t];
            S[3] = (1.0/6.0)*S0*sddd[t];  S2[3] = (1.0/3.0)*S20*sddd[t];
        }
        else { // Last value is unconditional.
            dtp1 = 0.0;
            p_set(E1,  mu0[t] - x0[t],  0.0, 0.0, 0.0);
            p_set(b,   b0[t] - x0[t],   0.0, 0.0, 0.0);
            p_set(S,   S0,              0.0, 0.0, 0.0);
            p_set(S2,  S20,             0.0, 0.0, 0.0);
        }

        // Compute polynomial delta, difference between conditional mean and mode
        p_subtract(delta, E1, b);
        p_set_scalar_mult(b_2, 2.0, b);

        // Compute V1 = Var[e_t|e_{t+1}], E[e_t|e_{t+1}]^2 and E2 = E[e_t^2|e_{t+1}]
        memcpy(V1, S, p_len * sizeof(double));
        p_square(E12, E1);
        p_add(E2, V1, E12);

        // Computation of C12 = Cov[e_t, e_t^2|e_{t+1}]
        p_set(C12, 4.0*delta[0]*S[0], 4.0*(delta[1]*S[0] + delta[0]*S[1]), 0.0, 0.0);
        p_mult_add(C12, b_2, V1);

        // Computation of V2 = Var[e_t^2|e_{t+1}]
        p_set_scalar_mult(V2, 2.0, S2);
        p_mult_add(V2, b_2, C12);

        memcpy(E3, C12, p_len * sizeof(double));
        p_mult_add(E3, E1, E2);

        // Part 2: compute m_t^{(i)}(e_{t+1}) and c_t^{(i,j)}(e_{t+1}) polynomials
        // in sequential procedure
        // -----------------------------------------------------------------------

        // Compute m_t^{(i)}(e_{t+1}) for quadratic forms
        for (iQ = 0; iQ < nQ; iQ++) {
            Q_term *Qi = &(Q[iQ]);

            // Add terms for e_t and e_t^2 in z^{(i)}(e_t, e_{t+1}) to
            // \tilde{m}^{(i)}_{t-1} and compute all terms of \tilde{m}^{(i)}_t
            // except the one with the expectation of e_t e_{t+1} 
            if (iQ < 3) {
                double Q_tt = ((t==0) || (t==n-1)) ? Qi->Q_11 : Qi->Q_tt;
                Qi->m_tm1[1] += Q_tt * dt_2 + Qi->Q_ttp * (dtm1 + dtp1);
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
        //g_cum[t] = Q[0].m_t[0];
        //g_cum[t+n] = Q[1].m_t[0];
        //g_cum[t+2*n] = Q[3].m_t[0];

        // Compute c_t^{(i,j)}(e_{t+1}) polynomials
        for (iC = 0; iC < nC; iC++) {
            C_term *Ci = &(C[iC]);
            Q_term *Qi = &(Q[Ci->i]), *Qj = &(Q[Ci->j]);

            // c_t^{(i,j)} = E[c_{t-1}^{i,j} | e_{t+1}] ...
            p_expect(Ci->c_t, Ci->c_tm1, E1, E2, E3);
            // ... + Cov[m_{t-1}^{(i)} + <z_i>, m_{t-1}^{(j)} + <z_j> | e_{t+1}],
            // where <z_i> is z_t^{(i)} less the e_t e_{t+1} term, same for j
            p_cov(Ci->c_t, Qi->m_tm1, Qj->m_tm1, V1, C12, V2);

            // Add term for e_t e_{t+1} product in tridiagonal cases
            if (iC < 5 && t < n-1) { // Qi is a tridiagonal quadratic form
                double V1_e = Qi->Q_ttp * Qj->m_tm1[1];    // Coeff of e_{t+1} var[e_t]
                double C12_e = Qi->Q_ttp * Qj->m_tm1[2];   // Coeff of e_{t+1} cov[e_t, e_t^2]
                if (iC < 3) { // Qj is a tridiagonal quadratic form
                    double V1_e2 = Qi->Q_ttp * Qj->Q_ttp;  // Coeff of e_{t+1} var[e_t^2]
                    V1_e += Qi->m_tm1[1] * Qj->Q_ttp;
                    C12_e += Qi->m_tm1[2] * Qj->Q_ttp;
                    Ci->c_t[2] += V1_e2 * V1[0];
                    Ci->c_t[3] += V1_e2 * V1[1];
                }
                Ci->c_t[1] += V1_e * V1[0] + C12_e * C12[0];
                Ci->c_t[2] += V1_e * V1[1] + C12_e * C12[1];
                Ci->c_t[3] += V1_e * V1[2] + C12_e * C12[2];
            }
            memcpy(Ci->c_tm1, Ci->c_t, p_len * sizeof(double));
        }
        dtm1 = dt;
        dt = dtp1;
        for (iQ = 0; iQ < nQ; iQ++)
            memcpy(Q[iQ].m_tm1, Q[iQ].m_t, p_len * sizeof(double));
    } // for(t=0; t<n-1; t++)

    // To avoid double counting
    *dt_sum -= d1;
    *dtt_sum -= d1 * d1;
    for (iQ = 0; iQ < 3; iQ++) {
        // Compute constant part of quadratic forms
        Q[iQ].dQd = Q[iQ].Q_11 * (d1*d1 + dn*dn);
        Q[iQ].dQd += Q[iQ].Q_tt * *dtt_sum + Q[iQ].Q_ttp * *dttp_sum;
    }
    // ... and linear forms.
    for (iQ=3; iQ<nQ; iQ++) {
        Q[iQ].qd = Q[iQ].q_1 * (d1 + dn) + Q[iQ].q_t * *dt_sum;
    }

    // Flat indices for a 3 x 3 matrix, 2 x 2 matrix
    // 0 1 2   0 1
    // 3 4 5   2 3
    // 6 7 8

    // Flat indices for a 3 x 3 x 3 array, 2 x 2 x 2 array
    // 0 1 2   9 10 11  18 19 20    0 1  4 5
    // 3 4 5  12 13 14  21 22 23    2 3  6 7
    // 6 7 8  15 16 17  24 25 26

    // Assign elements of expected gradient
    grad[0] = 0.5*n - 0.5 * omega * (Q[0].dQd + Q[0].m_t[0]); 
    grad[1] = -phi - 0.5 * omega * (Q[1].dQd + Q[1].m_t[0]);
    if (long_th) {
        grad[2] = omega * (Q[3].qd + Q[3].m_t[0]);
    }
    cov_Q1Q2[0] = 0.25 * omega * omega * C[6].c_t[0];

    // Assign elements of expected Hessian "Hess" and variance of gradient "var"
    Hess[0] = -0.5 * omega * (Q[0].dQd + Q[0].m_t[0]);
    var[0] = 0.25 * omega * omega * C[0].c_t[0];
    T[0] = Hess[0] + 3*var[0];
    if (long_th) {
        Hess[1] = Hess[3] = -0.5 * omega * (Q[1].dQd + Q[1].m_t[0]);
        Hess[4] = -(1-phi*phi) - 0.5 * omega * (Q[2].dQd + Q[2].m_t[0]);
        Hess[2] = Hess[6] = omega * (Q[3].qd + Q[3].m_t[0]);
        Hess[5] = Hess[7] = omega * (Q[4].qd + Q[4].m_t[0]);
        Hess[8] = -omega * (1-phi) * (n * (1-phi) + 2*phi);

        var[1] = var[3] = 0.25 * omega * omega * C[1].c_t[0];
        var[4] = 0.25 * omega * omega * C[2].c_t[0];
        var[2] = var[6] = -0.5 * omega * omega * C[3].c_t[0];
        var[5] = var[7] = -0.5 * omega * omega * C[4].c_t[0];
        var[8] = omega * omega * C[5].c_t[0];

        T[1] = T[3] = T[9] = Hess[1] + 3*var[1];
        T[4] = T[10] = T[12] = Hess[4] + 2*var[4];
        T[13] = -2*(3*phi*phi+1)*Hess[1] - 6*phi*Hess[4] + 0.25 * omega * omega * C[6].c_t[0];
        T[14] = -2*(3*phi*phi+1)*Hess[1];
        T[15] = -6*phi*Hess[4];
        T[16] = 0.25 * omega * omega * C[6].c_t[0];
    }
    else {
        Hess[1] = Hess[2] = -0.5 * omega * (Q[1].dQd + Q[1].m_t[0]);
        Hess[3] = -(1-phi*phi) - 0.5 * omega * (Q[2].dQd + Q[2].m_t[0]);

        var[1] = var[2] = 0.25 * omega * omega * C[1].c_t[0];
        var[3] = 0.25 * omega * omega * C[2].c_t[0];

        T[1] = T[2] = T[4] = Hess[1] + 3*var[1];
        T[3] = T[5] = T[6] = Hess[3] + 2*var[3];
        T[7] = -2*(3*phi*phi+1) * Hess[1] - 6*phi*Hess[3] + 0.25 * omega * omega * C[6].c_t[0];
    }
}
