#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "mex.h"
#include "RNG.h"
#include "symmetric_Hermite.h"


Symmetric_Hermite *symmetric_Hermite_alloc(int max_K, int max_n_reject)
{
    int k;
    Symmetric_Hermite *sh = (Symmetric_Hermite *) mxMalloc(sizeof(Symmetric_Hermite));
    sh->max_K = max_K;
    sh->max_n_reject = max_n_reject;
    sh->chi2_const = (double *) mxMalloc(max_K * sizeof(double));
    sh->cum_w = (double *) mxMalloc(max_K * sizeof(double));
    sh->coeff = (double *) mxMalloc(max_K * sizeof(double));
    sh->neg_coeff = (double *) mxMalloc(max_K * sizeof(double));
    
    for( k=0; k<max_K; k++ ) {
        double arg = 0.5 + k;
        sh->chi2_const[k] = exp(arg * log(2.0)) * tgamma(arg);
    }
    return sh;
}

void symmetric_Hermite_draw(Symmetric_Hermite *sh, double *x, double *log_f)
{
    int k, K = sh->K;     // Component index, number of components
    double y;             // x^2 proposal
    double f_poly, f_env_poly;
    
    // Compute component weights
    sh->c = 0.0;
    double c_plus = 0.0;
    for( k=0; k<K; k++ ) {
        double w = sh->coeff[k] * sh->chi2_const[k];
        sh->c += w;
        if( w >= 0 ) {
            c_plus += w;
            sh->neg_coeff[k] = 0.0;
        }
        else
            sh->neg_coeff[k] = sh->coeff[k];
        sh->cum_w[k] = c_plus;
    }
    sh->c_inv = 1.0 / sh->c;
    
    // Rejection sampling
    for(sh->n_reject=0; sh->n_reject<sh->max_n_reject; sh->n_reject++) {
        // Select component k according to w[k] weights
        double w = rng_rand() * sh->cum_w[K-1];
        if( w < sh->cum_w[0] ) // Quick discovery of main term
            k = 0;
        else {
            int lo, mid;
            for( lo=-1, k=K-1, mid=2; k > lo+1; mid = (lo + k)/2 ) {
                if( w >= sh->cum_w[mid] )
                    lo = mid;
                else
                    k = mid;
            }
        }
        
        // Propose y = x^2, accept/reject
        y = rng_chi2( 2*k + 1 );
        f_poly = poly_eval( sh->coeff, K, y );
        f_env_poly = f_poly - poly_eval( sh->neg_coeff, K, y );
        if (rng_rand() * f_env_poly < f_poly)
            break;
    }
    
    // Compute draw and density
    *log_f = -0.5 * y + log(f_poly * sh->c_inv);
    if (rng_rand() < 0.5)
        *x = sqrt(y);
    else
        *x = -sqrt(y);
}

double symmetric_Hermite_log_f( Symmetric_Hermite *sh, double x )
{
    int k, K = sh->K;     // Component index, number of components
    double y = x*x;
    sh->c = 0.0;
    for( k=0; k<K; k++ ) {
        double w = sh->coeff[k] * sh->chi2_const[k];
        sh->c += w;
    }
    sh->c_inv = 1.0/sh->c;
    double f_poly = poly_eval( sh->coeff, K, y );
    return -0.5 * y + log(f_poly * sh->c_inv);
}
