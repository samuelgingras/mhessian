#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "state.h"
#include "errors.h"


// Initial observation model
extern Observation_model plain_SV;
extern Observation_model student_SV;
extern Observation_model student_SV_AR;
extern Observation_model Poisson_SS;
extern Observation_model gammaPoisson_SS;
extern Observation_model exp_SS;

// Additional observation model
extern Observation_model mixture_SV;
extern Observation_model student_sd_SV;
extern Observation_model exp_SCD;
extern Observation_model weibull_SS;
extern Observation_model gamma_SS;
extern Observation_model gen_gamma_SS;
extern Observation_model regime_mix_exp_SS_cont;
extern Observation_model regime_mix_exp_SS_disc;

Observation_model *assign_model(const mxArray *prhs)
{
    mxArray *tmp = mxGetField(prhs, 0, "name");
    ErrMsgTxt( tmp != NULL,
    "Invalid input argument: model struct: 'name' missing");
    
    int name_length = mxGetN(tmp) + 1;
    char *name = (char *) mxMalloc( name_length * sizeof(char) );
    
    Observation_model *model;
    
    if( mxGetString(tmp, name, (mwSize)name_length) )
    {
        mexErrMsgTxt("Error assigning model \n");
    }
    else if( !strcmp(name, "plain_SV") )
    {
        model = &plain_SV;
    }
    else if( !strcmp(name, "mixture_SV") )
    {
        model = &mixture_SV;
    }
    else if( !strcmp(name, "student_SV") )
    {
        model = &student_SV;
    }
    else if( !strcmp(name, "student_sd_SV") )
    {
        model = &student_sd_SV;
    }
    else if( !strcmp(name, "student_SV_AR") )
    {
        model = &student_SV_AR;
    }
    else if( !strcmp(name, "Poisson_SS") )
    {
        model = &Poisson_SS;
    }
    else if( !strcmp(name, "gammaPoisson_SS") )
    {
        model = &gammaPoisson_SS;
    }
    else if( !strcmp(name, "exp_SS") )
    {
        model = &exp_SS;
    }
    else if( !strcmp(name, "exp_SCD") )
    {
        model = &exp_SCD;
    }
    else if( !strcmp(name, "gen_gamma_SS") )
    {
        model = &gen_gamma_SS;
    }
    else if( !strcmp(name, "gamma_SS") )
    {
        model = &gamma_SS;
    }
    else if( !strcmp(name, "weibull_SS") )
    {
        model = &weibull_SS;
    }
    else if( !strcmp(name, "regime_mix_exp_SS_cont") )
    {
        model = &regime_mix_exp_SS_cont;
    }
    else if( !strcmp(name, "regime_mix_exp_SS_disc") )
    {
        model = &regime_mix_exp_SS_disc;
    }
    else
    {
        mexErrMsgTxt("Unavailable model \n");
    }
    
    mxFree(name);
    return model;
}