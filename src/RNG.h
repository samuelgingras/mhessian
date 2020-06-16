#ifndef MEX_HESSIAN_RNG
#define MEX_HESSIAN_RNG


/* To initialize the Mersenne Twister Number Generator */
void rng_init_rand( unsigned long s );
void rng_init_by_array(unsigned long init_key[], int key_length);

/* Efficient Power operation */
double int_pow(double base, int exp);

/* Random Number Generator */
double rng_rand( void );					// U(0,1) open interval
double rng_gaussian( void );				// N(0,1)
double rng_gamma( double a, double b );
double rng_exp( double lambda );
double rng_chi2( double v );
double rng_t( double k );
double rng_beta( double a, double b );
int rng_binomial( double p, int n );
int rng_poisson( double mu );
int rng_n_binomial( double p, double n );


/* Probability density function */
double rng_chi2_pdf (double x, double nu);

/* Polynom evaluation: Homer's Method */
double poly_eval(double *coeffs, int s, double x);

#endif
