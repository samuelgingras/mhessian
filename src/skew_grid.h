#ifndef SKEWGRID
#define SKEWGRID

// Structure to hold grid values that never change
typedef struct {
  int K;
  double K_inv;
  double log_K;
  double *u;
  double *v;
  double *x;
  double *xu;
  double *c;
  double *cu;
  double *f_v;
  double *f_v_inv;
  double *Kf_v2_inv;
  double *f_v_prime;
  double *x_plus;
  double *x_minus;
  double *xu_plus;
  double *xu_minus;
} Grid;

#endif // SKEWGRID
