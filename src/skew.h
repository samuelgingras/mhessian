
#ifndef MEX_SKEW
#define MEX_SKEW

#define n1_max 10
#define n2_max 6

#include "symmetric_Hermite.h"

typedef struct {
	double mode;
	double h2;
	double h3;
	double h4;
	double h5;
	double s2_prior;
	double u_sign;
	int is_draw;
	double z;
  int n_reject;
	double log_density;
} Skew_parameters;

// Structure describing coefficients a3, a4, a5 and approximation parameters
// derived from them
typedef struct {
	double a2, a3, a4, a5;
	double p1[n1_max+1];
	double p2[n2_max+1];
	double p[n1_max+n2_max+1];
	int n1;
	int n2;
	double x;
	double g;
	double ge;
	double go;
} Skew_Approximation;

void skew_draw_parameter_string(Skew_parameters *skew, char *msg);
void skew_draw_eval( Skew_parameters *skew, Symmetric_Hermite *sh,
                    double K_1_threshold, double *K_2_threshold );

#endif
