#ifndef SPLINE
#define SPLINE

// For internal C use

void spline_eval(int n_knots, double *p, double *m, int n_evals, double *u, double *f_u);
void spline_draw(int n_knots, double *p, double *m, int n_draws, double *u);
double inner_t_draw(double p_k, double m_k);
double left_t_draw(double p_0, double m_0);
double right_t_draw(double p_K, double m_K);

#endif /* SPLINE */
