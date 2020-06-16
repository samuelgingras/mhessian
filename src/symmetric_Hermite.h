#ifndef MEX_SYMMETRIC_HERMITE
#define MEX_SYMMETRIC_HERMITE

typedef struct {
    int max_K;          // Maximum number of terms
    int n_reject;       // Number of iterations used
    int max_n_reject;   // Maximum number of iterations
    int K;              // Current number of terms
    double *chi2_const; // Chi squared normalization constants
    double *coeff;      // Polynomial coefficients
    double *neg_coeff;  // Negative coefficients min(coeff[k],0)
    double c;           // Integration constant
    double c_inv;       // Reciprocal of c
    double *cum_w;      // Cumulative probability weights
} Symmetric_Hermite;

Symmetric_Hermite *symmetric_Hermite_alloc(int max_K, int max_n_reject);
void symmetric_Hermite_draw(Symmetric_Hermite *sh, double *x, double *log_f);
double symmetric_Hermite_log_f(Symmetric_Hermite *sh, double x);

#endif
