%hessianMethod Highly Efficient Simulation Smoothing (In A Nutshell).
%   HMOUT = hessianMethod(MODEL,DATA,THETA) returns a structure HMOUT with
%   fields
%   * x, a draw from the HESSIAN approximation of the conditional posterior
%   of the state vector;
%   * xC, the mode of the conditional posterior of the state vector, also
%   equal to the mode of the HESSIAN approximation of this distribution;
%   * lnp_y__x, the log conditional density of observed data given the
%   state vector;
%   * lnp_x, the log conditional density of the state vector, given
%   parameters mu, phi and omega;
%   * lnq_x__y, the log density of the HESSIAN approximation, evaluated at
%   x
%   * MODEL is one of
%       - 'gaussian_SV', 'mix_gaussian_SV', 'student_SV'
%       - 'poisson_SS', 'gammapoisson_SS'
%       - 'exp_SS', 'gamma_SS', 'weibull_SS', 'gengamma_SS',
%       - 'burr_SS', 'mix_exp_SS', 'mix_gamma_SS'
%   * DATA is a sample from the given model
%   * THETA is a vector with fields
%       - N, the number of observations
%       - mu, the mean of the state vector
%       - phi, the autocorrelation of the state vector
%       - omega, the precision of the innovation of the state vector
% 
%   [HMOUT, STATE] = hessianMethod(MODEL,DATA,THETA) also returns a structure 
%   with various derivative evaluations and diagnostics variables.
% 
%   [...] = hessianMethod(...,'EvalAtState',X) evaluates the log density of the
%   HESSIAN approximation of the conditional posterior distribution of the state
%   vector, at the value X.
% 
%   [...] = hessianMethod(...,'EvalAtMode',TRUE) evaluates the log density of
%   the HESSIAN approximation of the conditional posterior distribution of
%   the state vector, at the modal value.
% 
%   [...] = hessianMethod(...,'GradHess',TYPE) returns, in addition, an
%   approximation of the gradient and Hessian of the log posterior density of
%   theta (with the state vector marginalized out). The vector theta, for given
%   value of TYPE is as follows:
%               'Long'  for (sigma, phi, mu)
%              'Short'  for (sigma, phi)
% 
%   See also evalState, evalObs, drawState, drawObs
