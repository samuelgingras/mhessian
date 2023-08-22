#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "state.h"
#include "model.h"

// Stochastic Volatility models
extern Observation_model gaussian_SV, student_SV;

// Dynamic count models
extern Observation_model poisson_SS, gammapoisson_SS;

// Multiplicative Error models
extern Observation_model exp_SS, gamma_SS, weibull_SS, gengamma_SS, burr_SS;

// Finite mixture models
extern Observation_model mix_gaussian_SV, mix_exp_SS, mix_gamma_SS;

static Observation_model *model_list[] = {
    &gaussian_SV, &student_SV,
    &poisson_SS, &gammapoisson_SS,
    &exp_SS, &gamma_SS, &weibull_SS, &gengamma_SS, &burr_SS,
    &mix_gaussian_SV, &mix_exp_SS, &mix_gamma_SS
};
static int n_models = sizeof(model_list)/sizeof(Observation_model *);

Observation_model *findModel(char *model_name)
{
    Observation_model *model = NULL;
    for (int i=0; i<n_models; i++)
        if (!strcmp(model_name, model_list[i]->name))
            return model_list[i];
    return NULL;
}
