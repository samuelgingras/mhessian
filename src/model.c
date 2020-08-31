#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "state.h"
#include "errors.h"

extern Observation_model gaussian_SV;
extern Observation_model mix_gaussian_SV;
extern Observation_model student_SV;
extern Observation_model poisson_SS;
extern Observation_model gammapoisson_SS;
extern Observation_model exp_SS;
extern Observation_model gamma_SS;
extern Observation_model weibull_SS;
extern Observation_model gengamma_SS;
extern Observation_model mix_exp_SS;
extern Observation_model mix_gamma_SS;
extern Observation_model regime_mix_exp_SS;
extern Observation_model regime_mix_exp_SS_tobs;

Observation_model *assignModel(const mxArray *prhs)
{
    char *name = NULL;
    Observation_model *model;

    if( mxIsStruct(prhs) )
    {
        mxArray *tmp = mxGetField( prhs, 0, "name" );
        if( tmp != NULL && mxIsChar(tmp) )
            name = mxArrayToString( tmp );
        else
            mexErrMsgIdAndTxt( "mhessian:missingInputs",
                "Model: Field 'name' required." );
    }
    else if( mxIsChar(prhs) )
        name = mxArrayToString( prhs );
    else
        mexErrMsgIdAndTxt( "mhessian:missingInputs",
            "Model name required.");

    if( name == NULL )
        mexErrMsgIdAndTxt( "mhessian:hessianMethod:readingFailed",
            "Error reading model name." );


    if( !strcmp(name, "gaussian_SV") )
    {
        model = &gaussian_SV;
    }
    else if( !strcmp(name, "mix_gaussian_SV") )
    {
        model = &mix_gaussian_SV;
    }
    else if( !strcmp(name, "student_SV") )
    {
        model = &student_SV;
    }
    else if( !strcmp(name, "poisson_SS") )
    {
        model = &poisson_SS;
    }
    else if( !strcmp(name, "gammapoisson_SS") )
    {
        model = &gammapoisson_SS;
    }
    else if( !strcmp(name, "exp_SS") )
    {
        model = &exp_SS;
    }
    else if( !strcmp(name, "mix_exp_SS") )
    {
        model = &mix_exp_SS;
    }
    else if( !strcmp(name, "mix_gamma_SS") )
    {
        model = &mix_gamma_SS;
    }
    else if( !strcmp(name, "gamma_SS") )
    {
        model = &gamma_SS;
    }
    else if( !strcmp(name, "weibull_SS") )
    {
        model = &weibull_SS;
    }
    else if( !strcmp(name, "gengamma_SS") )
    {
        model = &gengamma_SS;
    }
    else if( !strcmp(name, "regime_mix_exp_SS") )
    {
        model = &regime_mix_exp_SS;
    }
    else if( !strcmp(name, "regime_mix_exp_SS_tobs") )
    {
        model = &regime_mix_exp_SS_tobs;
    }
    else
    {
        mexErrMsgIdAndTxt( "mhessian:invalidInputs",
            "Observation model not available.");
    }
    
    mxFree(name);
    return model;
}