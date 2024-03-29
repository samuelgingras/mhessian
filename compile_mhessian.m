% Folder Organization
DIRECTORY = pwd;
CALL = fullfile(pwd,'/mex');
MODEL_LOCATION = fullfile(pwd,'/model');
COMPUTATION_LOCATION = fullfile(pwd,'/src');

% TODO: add message for this part
% Add mex directory to path
path(CALL, path)

% TODO: add options for this part i.e. verbose true/false
% Compilation options
IPATH = ['-I' COMPUTATION_LOCATION];
MEXOPTS = {'-R2018a', '-DMEX', '-outdir', CALL, IPATH};

%OBSERVATION_MODEL = { ...
%    fullfile(COMPUTATION_LOCATION,'model.c'), ...
%    fullfile(COMPUTATION_LOCATION,'matlab_misc.c'), ...
%    fullfile(MODEL_LOCATION,'weibull_SS.c'), ...
%    fullfile(MODEL_LOCATION,'mix_gamma_SS.c'), ...
%};

OBSERVATION_MODEL = { ...
    fullfile(COMPUTATION_LOCATION,'model.c'), ...
    fullfile(COMPUTATION_LOCATION,'matlab_misc.c'), ...
    fullfile(MODEL_LOCATION,'gaussian_SV.c'), ...
    fullfile(MODEL_LOCATION,'mix_gaussian_SV.c'), ...
    fullfile(MODEL_LOCATION,'gamma_SS.c'), ...
    fullfile(MODEL_LOCATION,'exp_SS.c'), ...
    fullfile(MODEL_LOCATION,'gammapoisson_SS.c'), ...
    fullfile(MODEL_LOCATION,'gengamma_SS.c'), ...
    fullfile(MODEL_LOCATION,'student_SV.c'), ...
    fullfile(MODEL_LOCATION,'poisson_SS.c'), ...
    fullfile(MODEL_LOCATION,'mix_exp_SS.c'), ...
    fullfile(MODEL_LOCATION,'burr_SS.c'), ...
};

X_UNIVARIATE = { ...
    fullfile(COMPUTATION_LOCATION,'x_univariate.c'), ...
    fullfile(COMPUTATION_LOCATION,'Phi.c'), ...
    fullfile(COMPUTATION_LOCATION,'alias.c'), ...
    fullfile(COMPUTATION_LOCATION,'spline.c') ...
    fullfile(COMPUTATION_LOCATION,'skew.c'), ...
    fullfile(COMPUTATION_LOCATION,'skew_grid.c'), ...
    fullfile(COMPUTATION_LOCATION,'skew_spline.c'), ...
    fullfile(COMPUTATION_LOCATION,'symmetric_Hermite.c'), ...
    fullfile(COMPUTATION_LOCATION,'state.c'), ...
    fullfile(COMPUTATION_LOCATION,'RNG.c'), ...
    fullfile(COMPUTATION_LOCATION,'errors.c'), ...
    fullfile(COMPUTATION_LOCATION,'grad_hess.c')
};

OBSERVATION_DRAW = { ...
    fullfile(COMPUTATION_LOCATION,'errors.c'), ...
    fullfile(COMPUTATION_LOCATION,'state.c'), ...
    fullfile(COMPUTATION_LOCATION,'RNG.c') ...
};

% Compile mex files
mex(MEXOPTS{:}, 'mex/hessianMethod.c', OBSERVATION_MODEL{:}, X_UNIVARIATE{:})
mex(MEXOPTS{:}, 'mex/drawState.c', X_UNIVARIATE{:})
mex(MEXOPTS{:}, 'mex/evalState.c', X_UNIVARIATE{:})
mex(MEXOPTS{:}, 'mex/drawObs.c', OBSERVATION_MODEL{:}, OBSERVATION_DRAW{:})
mex(MEXOPTS{:}, 'mex/evalObs.c', OBSERVATION_MODEL{:}, OBSERVATION_DRAW{:})

% Reset directory
cd(DIRECTORY)
clear DIRECTORY CALL MODEL_LOCATION COMPUTATION_LOCATION IPATH MEXOPTS
clear OBSERVATION_MODEL X_UNIVARIATE OBSERVATION_DRAW