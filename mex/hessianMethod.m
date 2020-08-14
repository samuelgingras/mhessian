%hessianMethod Highly efficient simulation smoothing (in a Nutshell).
%   HMOUT = hessianMethod(MODEL,DATA,THETA)
% 
%   [HMOUT, STATE] = hessianMethod(MODEL,DATA,THETA) returns the structure 
%   of derivatives.
% 
%   [...] = hessianMethod(...,'EvalAtState',X) evaluate the HESSIAN
%   approximation at the state vector X.
% 
%   [...] = hessianMethod(...,'EvalAtMode',TRUE) evaluate the HESSIAN
%   approximation at the mode.
% 
%   [...] = hessianMethod(...,'GradHess',TYPE) return in addition the 
%   Gradient and Hessian approximation of the given TYPE.
%               'Long'  For (sigma, phi, mu)
%              'Short'  For (sigma, phi)
% 