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
MEXOPTS = {'-R2018a','-DMEX','-outdir',CALL,IPATH};

OBSERVATION_MODEL = {fullfile(COMPUTATION_LOCATION,'model.c'), ...
                     fullfile(COMPUTATION_LOCATION,'faa_di_bruno.c'), ...
                     fullfile(MODEL_LOCATION,'gaussian_SV.c'), ...
                     fullfile(MODEL_LOCATION,'student_sd_SV.c')};

ALPHA_UNIVARIATE = {fullfile(COMPUTATION_LOCATION,'alpha_univariate.c'), ...
                    fullfile(COMPUTATION_LOCATION,'Phi.c'), ...
                    fullfile(COMPUTATION_LOCATION,'alias.c'), ...
                    fullfile(COMPUTATION_LOCATION,'spline.c') ...
                    fullfile(COMPUTATION_LOCATION,'skew.c'), ...
                    fullfile(COMPUTATION_LOCATION,'skew_grid.c'), ...
                    fullfile(COMPUTATION_LOCATION,'skew_spline.c'), ...
                    fullfile(COMPUTATION_LOCATION,'symmetric_Hermite.c'), ...
                    fullfile(COMPUTATION_LOCATION,'state.c'), ...
                    fullfile(COMPUTATION_LOCATION,'RNG.c'), ...
                    fullfile(COMPUTATION_LOCATION,'errors.c')};

OBSERVATION_DRAW = {fullfile(COMPUTATION_LOCATION,'errors.c'), ...
                    fullfile(COMPUTATION_LOCATION,'state.c'), ...
                    fullfile(COMPUTATION_LOCATION,'RNG.c')};

% Compile mex files (new interface)
mex(MEXOPTS{:}, 'hessianMethod.c', OBSERVATION_MODEL{:}, ALPHA_UNIVARIATE{:})
mex(MEXOPTS{:}, 'drawState.c', ALPHA_UNIVARIATE{:})
mex(MEXOPTS{:}, 'evalState.c', ALPHA_UNIVARIATE{:})
mex(MEXOPTS{:}, 'drawObs.c', OBSERVATION_MODEL{:}, OBSERVATION_DRAW{:})
mex(MEXOPTS{:}, 'evalObs.c', OBSERVATION_MODEL{:}, OBSERVATION_DRAW{:})

% % TODO: add to options
% mex(MEXOPTS{:}, 'fct/grad_hess_approx.c', 'src/grad_hess.c')

% Reset directory
cd(DIRECTORY)
clear DIRECTORY CALL MODEL_LOCATION COMPUTATION_LOCATION IPATH MEXOPTS
clear OBSERVATION_MODEL ALPHA_UNIVARIATE OBSERVATION_DRAW