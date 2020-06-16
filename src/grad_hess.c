#include <math.h>
#include "mex.h"

static inline void poly_mult(double *poly, const double *poly1, const double *poly2)
{
    poly[0] = poly1[0]*poly2[0];
    poly[1] = poly1[0]*poly2[1] + poly1[1]*poly2[0];
    poly[2] = poly1[0]*poly2[2] + poly1[1]*poly2[1] + poly1[2]*poly2[0];
}

static inline void poly_copy(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] = poly1[i];
}

static inline void poly_add(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] += poly1[i];
}

static inline void poly_subtract(double *poly, const double *poly1)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] -= poly1[i];
}

static inline void poly_scalar_mult(double *poly, double c)
{
    int i;
    for (i=0; i<3; i++)
        poly[i] *= c;
}

static inline void poly_recip_1m(double *poly, double *poly1)
{
    poly[0] = poly1[0] * (1 + poly1[0]);
    poly[1] = poly1[1] * (1 + 2*poly1[0]);
    poly[2] = poly1[1] * poly1[1] + poly1[2] * (1 + 2*poly1[0]);
}

static inline double poly_eval_mean(double *poly, double Vareps_tp1)
{
    return poly[0] + poly[2]*Vareps_tp1;
}

static inline void poly_print(char *s, double *poly)
{
    mexPrintf("%s: %lf + %lf e + %lf e^2\n", s, poly[0], poly[1], poly[2]);
}

static inline double *mxStateGetPr( const mxArray *mxState, char *field_name)
{
    mxArray *field_pr = mxGetField(mxState,0,field_name);
    return mxGetPr(field_pr);
}

void compute_grad_Hess(
    const mxArray *mxState,
    int n,
    double *mu,
    double phi,
    double omega,
    double *u,
    double *grad,
    double *Hess,
    double *var
    )
{
    int i, j, t, vec_type;
    double *x0 = mxStateGetPr(mxState,"x_mode");
    double *Sigma = mxStateGetPr(mxState,"Sigma");
    double *ad = mxStateGetPr(mxState,"ad");
    double *add = mxStateGetPr(mxState,"add");
    double *mu0 = mxStateGetPr(mxState,"mu");
    double *mud = mxStateGetPr(mxState,"mud");
    double *mudd = mxStateGetPr(mxState,"mudd");
    double *s0 = mxStateGetPr(mxState,"Sigma");
    double *sd = mxStateGetPr(mxState,"sd");
    double *b = mxStateGetPr(mxState,"b");
    double *bd = mxStateGetPr(mxState,"bd");
    double *bdd = mxStateGetPr(mxState,"bdd");
    double *bddd = mxStateGetPr(mxState,"bddd");
    double *psi = mxStateGetPr(mxState,"psi");
    int psi_stride = 6;     // Return as an element of State ?
    
    double *m = (double *) mxMalloc(n * sizeof(double));
    double *e = (double *) mxMalloc(n * sizeof(double));
    double *Sig_adj = (double *) mxMalloc(n * sizeof(double));
    double *mud_adj = (double *) mxMalloc(n * sizeof(double));
    
    double Eeps_t = 0.0, s2_t, Vareps_t = 0.0, const_t = 0.0;
    double Eeps_tp1, Vareps_tp1, Ee3_tp1, Ee4_tp1, Vare2_tp1, const_tp1;
    double Varu_t, Eu_teps_tp;
    
    double poly_delta[3], poly_delta2[3], poly_b[3], poly_b2[3], poly_mu[3];
    double poly_h_den[3], poly_h_den_2[3], poly_h_num[3];
    double h = omega*(1-phi*phi);
    double H = omega*(1-phi)*(1-phi);
    
    // Case n-1
    double *psi_t = psi + (n-1)*psi_stride;
    double b_n = b[n-1] - x0[n-1];
    
    double h_den = omega*phi*(mud[n-2] - ad[n-2] + mudd[n-2]*b_n + 0.5*bddd[n-2]*b_n*b_n);
    h_den += psi_t[3]*b_n + 0.5*psi_t[4]*b_n*b_n;
    s2_t = Sigma[n-1] * 1/(1-Sigma[n-1]*h_den);
    s2_t *= (1 + 0.5 * psi_t[4] * s2_t * s2_t);
    double delta_n = 0.5*s2_t*s2_t*(psi_t[3] + psi_t[4]*b_n + omega*phi*mudd[n-2] + bddd[n-1]*b_n*b_n);
    Eeps_t = b_n + delta_n;
    Eeps_t += 0.01;
    s2_t -= delta_n*delta_n;
    Sig_adj[n-1] = s2_t;
    Vareps_t = s2_t;
    double diag_tr = (1-phi*phi) * s2_t * s2_t;
    double Az_tm1 = diag_tr;
    double off_diag_tr = diag_tr;
    
    m[n-1] = const_t = x0[n-1] + Eeps_t - mu[n-1];
    grad[0] = (1-phi*phi) * (const_t * const_t + Vareps_t);
    grad[1] = phi * (const_t * const_t + Vareps_t);
    grad[2] = omega * (1-phi) * const_t;
    Hess[4] = 0.0;
    Hess[5] = const_t;
    
    // To carry to next iteration
    Ee3_tp1 = 5*s2_t*delta_n + 3*delta_n*delta_n*delta_n;
    Ee4_tp1 = 3*s2_t*s2_t;
    Vare2_tp1 = 2*s2_t*s2_t;
    
    e[n-1] = u[n-1] * sqrt(Sigma[n-1]);
    for (t=n-2; t>=0; t--) {
        double *psi_t = psi + t*psi_stride;
        
        // Values from t+1
        Eeps_tp1 = Eeps_t;
        Vareps_tp1 = Vareps_t;
        const_tp1 = const_t;
        
        // Compute polynomials mu(e_{t+1}), b(e_{t+1}) and their difference delta(e_{t+1})
        poly_mu[0] = (mu0[t] - x0[t]) + (mud[t] + (0.5*mudd[t] + (1.0/6)*bddd[t]*Eeps_tp1)*Eeps_tp1)*Eeps_tp1;
        poly_mu[1] = mud[t] + (mudd[t] + 0.5*bddd[t]*Eeps_tp1)*Eeps_tp1;
        poly_mu[2] = 0.5*(mudd[t] + bddd[t]*Eeps_tp1);
        poly_b[0] = (b[t] - x0[t]) + (bd[t] + (0.5*bdd[t] + (1.0/6)*bddd[t]*Eeps_tp1)*Eeps_tp1)*Eeps_tp1;
        poly_b[1] = bd[t] + (bdd[t] + 0.5*bddd[t]*Eeps_tp1)*Eeps_tp1;
        poly_b[2] = 0.5*(bdd[t] + bddd[t]*Eeps_tp1);
        poly_copy(poly_delta, poly_mu);
        poly_subtract(poly_delta, poly_b);
        
        // Compute E[eps_t] term
        Eeps_t = poly_eval_mean(poly_mu, Vareps_tp1) + (1.0/6)*bddd[t]*Ee3_tp1;
        
        // Compute s2_t
        poly_mult(poly_b2, poly_b, poly_b);
        poly_copy(poly_h_den, poly_b);
        poly_scalar_mult(poly_h_den, ad[t]*((t==0)?0.0:mudd[t-1]) + Sigma[t]*psi_t[3]);
        poly_copy(poly_h_den_2, poly_b2);
        poly_scalar_mult(poly_h_den_2, 0.5*(ad[t]*((t==0)?0.0:bddd[t-1]) + Sigma[t]*psi_t[4]));
        poly_add(poly_h_den, poly_h_den_2);
        if (t>0)
            poly_h_den[0] += ad[t] * (mud[t-1] - ad[t-1]);
        poly_recip_1m(poly_h_num, poly_h_den);
        poly_mult(poly_delta2, poly_delta, poly_delta);
        double eval1 = poly_eval_mean(poly_h_num, Vareps_tp1);
        double eval2 = poly_eval_mean(poly_delta2, Vareps_tp1);
        double s2_t = Sigma[t] * (1 + eval1);
        s2_t *= (1 + 0.5 * psi_t[4] * s2_t * s2_t);
        s2_t -= eval2;
        Sig_adj[t] = s2_t;
        mud_adj[t] = poly_mu[1];
        
        // Add Var[E[eps_t|eps_{t+1}] term to get Var[eps_t] term
        Vareps_t = s2_t + poly_mu[1]*poly_mu[1] * Vareps_tp1;
        Vareps_t += 2 * poly_mu[1] * poly_mu[2] * Ee3_tp1;
        Vareps_t += 2 * poly_mu[1] * (1.0/6) * bddd[t] * Ee4_tp1;
        Vareps_t += poly_mu[2] * poly_mu[2] * Vare2_tp1;
        
        // Var[E[e_t - e_{t+1}|x_{t+1}]]
        double mud_phi_t = poly_mu[1] - phi;
        Varu_t = s2_t + mud_phi_t*mud_phi_t * Vareps_tp1;
        Varu_t += 2 * mud_phi_t * poly_mu[2] * Ee3_tp1;
        Varu_t += 2 * mud_phi_t * (1.0/6) * bddd[t] * Ee4_tp1;
        Varu_t += poly_mu[2] * poly_mu[2] * Vare2_tp1;
        diag_tr += Varu_t * Varu_t;
        if (t>1) {
            double xx = Vareps_t - phi * poly_mu[1] * Vareps_tp1;
            Az_tm1 = poly_mu[1]*poly_mu[1]*Az_tm1 + xx*xx;
            off_diag_tr += mud_phi_t * mud_phi_t * Az_tm1;
        }
        
        Eu_teps_tp = mud_phi_t * Vareps_tp1;
        Eu_teps_tp += poly_mu[2] * Ee3_tp1;
        Eu_teps_tp += (1.0/6) * bddd[t] * Ee4_tp1;
        
        // Compute terms of gradient elements
        m[t] = const_t = x0[t] + Eeps_t - mu[t];
        double c1 = const_t - phi*const_tp1;
        double c2 = c1 * const_tp1;
        grad[0] += c1*c1 + Varu_t;
        grad[1] += c2 + Eu_teps_tp;
        if (t==0) {
            grad[2] += omega * (1-phi) * const_t;
            Hess[5] += const_t;
        }
        else {
            grad[2] += H * const_t;
            Hess[4] += const_t * const_t + Vareps_t;
            Hess[5] += 2 * (1-phi) * const_t;
        }
        
        // Compute e from common random numbers u
        e[t] = (mud[t] + 0.5*mudd[t]*e[t+1]) * e[t+1] + sqrt(Sigma[t]) * u[t];
        
        // Compute current values for next iteration: tp1 is the current value of t, to become t+1
        double Ee3 = Ee3_tp1;
        Ee3_tp1 = poly_mu[1]*poly_mu[1]*poly_mu[1]*Ee3 + 3*poly_mu[1]*poly_mu[1]*poly_mu[2]*Vare2_tp1;
        double delta_t = poly_eval_mean(poly_delta, Vareps_tp1);
        Ee3_tp1 += 5 * s2_t * delta_t + 3 * delta_t * delta_t * delta_t;
        Ee3_tp1 += 3*Sigma[t]
            * (poly_h_num[1]*poly_mu[1]*Vareps_tp1 + (poly_mu[1]*poly_h_num[2]+poly_mu[2]*poly_h_num[1])*Ee3);
        Vare2_tp1 = 2*Vareps_t*Vareps_t;
        Ee4_tp1 = 3*Vareps_t*Vareps_t;
    }
    Hess[0] = -0.5*omega*grad[0];
    grad[0] = Hess[0] + 0.5*n;
    grad[1] = grad[1] * h - phi;
    
    Hess[1] = Hess[3] = 0.0;
    Hess[2] = Hess[6] = 0.0;
    Hess[4] = -(1-phi*phi) * h * Hess[4] - (1-phi*phi);
    Hess[7] = Hess[5] = 0.0;
    Hess[8] = -n*H - 2*phi*omega*(1-phi);
    
    double mQSQm[2][2] = {0.0}, eQSQe[2][2] = {0.0};
    double Qm_t[2] = {0.0}, Qe_t[2] = {0.0};
    double wm_t[3] = {0.0}, we_t[2] = {0.0};
    double wm_tm1[3] = {0.0}, we_tm1[2] = {0.0};
    double mQSv[2] = {0.0};
    double vSv = 0.0;
    for (t=0; t<n; t++) {
        
        for (vec_type=0; vec_type<2; vec_type++) { // Same computations for m and e vectors
            double *v = vec_type ? m : e;
            double *Qv_t = vec_type ? Qm_t : Qe_t;
            double *w_t = vec_type ? wm_t : we_t;
            double *w_tm1 = vec_type ? wm_tm1 : we_tm1;
            double (*quad_form)[2] = vec_type ? mQSQm : eQSQe;
            
            if (t==0) {
                Qv_t[0] = v[t] - phi * v[t+1];
                Qv_t[1] = 0.5 * v[t+1];
            }
            else if (t==n-1) {
                Qv_t[0] = v[t] - phi * v[t-1];
                Qv_t[1] = 0.5 * v[t-1];
            }
            else {
                Qv_t[0] = (1+phi*phi) * v[t] - phi * (v[t-1] + v[t+1]);
                Qv_t[1] = -phi * v[t] + 0.5 * (v[t-1] + v[t+1]);
            }
            for (i=0; i<2; i++) {
                w_t[i] = Qv_t[i];
                if (t>0)
                    w_t[i] += mud_adj[t-1] * w_tm1[i];
            }
            if (vec_type) {
                w_t[2] = (t==0 || t==n-1) ? (1-phi) : (1-phi)*(1-phi);
                if (t>0)
                    w_t[2] += mud_adj[t-1] * w_tm1[2];
                for (i=0; i<2; i++)
                    mQSv[i] += Sig_adj[t] * w_t[i] * w_t[2];
                vSv += Sig_adj[t] * w_t[2] * w_t[2];
                w_tm1[2] = w_t[2];
            }
            for (i=0; i<2; i++) {
                for (j=0; j<=i; j++)
                    quad_form[i][j] += Sig_adj[t] * w_t[i] * w_t[j];
                w_tm1[i] = w_t[i];
            }
        }
    }

    var[0] = (2*(diag_tr + 2*off_diag_tr) + 4*mQSQm[0][0]) * 0.25 * omega * omega;
    var[1] = var[3] = (2*eQSQe[1][0] + 4*mQSQm[1][0]) * -0.5 * omega * h;
    var[4] = (2*eQSQe[1][1] + 4*mQSQm[1][1]) * h * h;
    var[2] = var[6] = 2*mQSv[0] * -0.5 * omega * omega;
    var[5] = var[7] = 2*mQSv[1] * omega * h;
    var[8] = vSv * omega * omega;

    mxFree(e);
    mxFree(m);
    mxFree(Sig_adj);
    mxFree(mud_adj);
}