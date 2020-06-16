#ifndef SKEW_SPLINE
#define SKEW_SPLINE

#include "skew_grid.h"

// For internal C use
void skew_spline_draw_eval(int is_draw, int n_grid_points, int is_v,
                          double mode, double *h, double mu, double omega,
                          double *z, double *ln_f);

#endif /* SKEW_SPLINE */
