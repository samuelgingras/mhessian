function [lnp_th, lnp_th_g, lnp_th_H] = log_prior_eval(prior, theta)

  if( isfield(theta,'x') )
    [lnp_th, lnp_th_g, lnp_th_H] = prior.log_prior_eval(prior.hyper, theta.x);
  else
    [lnp_th, lnp_th_g, lnp_th_H] = prior.log_prior_eval(prior.hyper, theta);
  end
  
end