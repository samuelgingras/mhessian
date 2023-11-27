% File: gaussian_sv_joint_draw.m
% 
% Date: 2023-10-03
% Author: William McCausland

% Clean workspace
%clear all
%clc

% Set seeds of random number generators
rng(1)
drawState(12)
drawObs(123)
hessianMethod(1234)

% Set simulation parameters
ndraw  = 20000;             % Number of posterior draws 
burnin = 2;                 % Number of burn-in draws 
ndata  = 5000;              % Number of artificial observations

% Set model, values of true parameters
model  = 'gaussian_SV';     % Observation model
mu0    = -9.5;              % Mean
phi0   = 0.97;              % Autocorrelation
omega0 = 25.0;              % Precision

% Set theta for simulating artificial data
theta.N = ndata;
theta.mu = mu0;
theta.phi = phi0;
theta.omega = omega0;

% Simulate artificial sample
x = drawState(theta);
y = drawObs(x, model, theta);

% Set prior distribution (mean and variance are arguments)
%prior = set_MVN_prior(true, [3.6; 2.5; -10.5], ...
%                      [1.25, 0.5, 0.0; 0.5, 0.25, 0.0; 0.0, 0.0, 0.25]);
%prior = set_GaBeN_prior(true, 1, 50, 19, 1, -9, 1);
prior = set_LNBeN_prior(true, -3, 1, 19, 1, -9, 1);

postsim = joint_draw_postsim(model, prior, y, ndraw, burnin);