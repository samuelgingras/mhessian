#include "faa_di_bruno.h"

void compute_Faa_di_Bruno( int n, double *f, double *g, double *fg )
{
	fg[0] = f[0];
	if( n >= 1) {
		fg[1] = f[1]*g[1];
		if( n >= 2 ) {
			double g1_2 = g[1]*g[1];
			fg[2] = f[1]*g[2] + f[2]*g1_2;
			if( n >= 3 ) {
				double g1_3 = g1_2*g[1];
				fg[3] = f[1]*g[3] + 3*f[2]*g[1]*g[2] + f[3]*g1_3;
				if( n >= 4 ) {
					double g2_2 = g[2]*g[2];
					double g1_4 = g1_3*g[1];
					fg[4] = f[1]*g[4] + 4*f[2]*g[1]*g[3]
						+ 3*f[2]*g2_2 + 6*f[3]*g1_2*g[2] + f[4]*g1_4;
					if( n >= 5 ) {
						double g1_5 = g1_4*g[1];
						fg[5] = f[1]*g[5] + 5*f[2]*g[1]*g[4] + 10*f[2]*g[2]*g[3]
							+ 15*f[3]*g2_2*g[1] + 10*f[3]*g[3]*g1_2 + 10*f[4]*g[2]*g1_3
							+ f[5]*g1_5;
					}
				}
			}
		}
	}
}