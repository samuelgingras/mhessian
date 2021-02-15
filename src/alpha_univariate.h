#ifndef MEX_ALPHA_UNIVARIATE
#define MEX_ALPHA_UNIVARIATE

#include "state.h"

void alpha_prior_draw(State_parameter *theta_alpha, double *alpha);
void alpha_prior_eval(State_parameter *theta_alpha, double *state, double *log_p);
int compute_alC(int trust_alC, int safe, Observation_model *model, Theta * theta, State *state, Data *data); 
void compute_alC_all(Observation_model *model, Theta *theta, State *state, Data *data);
void draw_HESSIAN(int isDraw, Observation_model *model,Theta *theta, State *state, Data *data, double *log_q );

#endif