#include "state.h"

#ifndef MODEL
#define MODEL

Observation_model *findModel(char *name);
void compute_Faa_di_Bruno(int n_derivs, double *f_derivs, double *g_derivs, double *fg_derivs);
int all_positive(Matrix *matrix);
int column_stochastic(Matrix *matrix);

#endif