function [omqiota, omqpiota, omqppiota]	= compute_omqiota(n, th2)
	omega = exp(th2(1));
	phi = tanh(th2(2));
	omqiota = omega * (1-phi) * ((n-2)*(1-phi) + 2);
	omqpiota = -omega * (1-phi^2) * (2*(n-2)*(1-phi) + 2);
	omqppiota = 2*omega * (1-phi^2) * ((n-2)*(1+3*phi)*(1-phi) + 2*phi);
end
