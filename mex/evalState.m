%evalState   Evaluation of the log density of the state vector, given parameters
%   LNP_X = evalState(X,THETA) calculates the log density of the state
%   vector at the value X, for given values of the parameters in THETA,
%   a structure with fields
%       - N, the number of observations and the length of X
%       - mu, the mean of the state vector
%       - phi, the autocorrelation of the state vector
%       - omega, the precision of the innovation of the state vector
%   and perhaps others, depending on the model.
%
%   See also hessianMethod, evalObs, drawState, drawObs
