#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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
    //&exp_SS, &gamma_SS, &weibull_SS, &gengamma_SS, &burr_SS,
    //&mix_gaussian_SV, &mix_exp_SS, &mix_gamma_SS
    &exp_SS, &gamma_SS, &gengamma_SS, &burr_SS,
    &mix_gaussian_SV, &mix_exp_SS
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

void compute_Faa_di_Bruno(int n, double *f, double *g, double *fg)
{
    fg[0] = f[0];
    if( n >= 1) {
        fg[1] = f[1]*g[1];
        if( n >= 2 ) {
            double g1_2 = g[1]*g[1];
            fg[2] = f[1]*g[2] + f[2]*g1_2;
            if( n >= 3 ) {
                double g1_3 = g1_2*g[1];
                fg[3] = f[1]*g[3] + 3*f[2]*g[1]*g[2] + f[3]*g1_3;
                if( n >= 4 ) {
                    double g2_2 = g[2]*g[2];
                    double g1_4 = g1_3*g[1];
                    fg[4] = f[1]*g[4] + 4*f[2]*g[1]*g[3]
                        + 3*f[2]*g2_2 + 6*f[3]*g1_2*g[2] + f[4]*g1_4;
                    if( n >= 5 ) {
                        double g1_5 = g1_4*g[1];
                        fg[5] = f[1]*g[5] + 5*f[2]*g[1]*g[4] + 10*f[2]*g[2]*g[3]
                            + 15*f[3]*g2_2*g[1] + 10*f[3]*g[3]*g1_2 + 10*f[4]*g[2]*g1_3
                            + f[5]*g1_5;
                    }
                }
            }
        }
    }
}

int all_positive(Matrix *matrix) {
    int n_elements = matrix->n_rows * matrix->n_cols;
    for (int i=0; i < n_elements; i++)
        if (matrix->p[i] < 0)
            return 0;
    return 1;
}

int column_stochastic(Matrix *matrix) {
    for (int col=0; col < matrix->n_cols; col++) {
        double sum = 0.0;
        for (int row=0; row < matrix->n_rows; row++)
            sum += matrix->p[row + col*matrix->n_rows];
        if (fabs(sum-1.0) > 1e-9)
            return 0;
    }
    return 1;
}

