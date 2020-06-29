#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mex.h"
#include "RNG.h"



/* Mersenne Twister Parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL       /* constant vector a */
#define UPPER_MASK 0x80000000UL     /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL     /* least significant r bits */

static unsigned long mt[N];         /* the array for the state vector  */
static int mti=N+1;                 /* mti==N+1 means mt[N] is not initialized */

/* Ziggurat Algorithm Parameter and Tables */
#define PARAM_R 3.44428647676

/* tabulated values for the height of the Ziggurat levels */
static const double ytab[128] = {
    1, 0.963598623011, 0.936280813353, 0.913041104253,
    0.892278506696, 0.873239356919, 0.855496407634, 0.838778928349,
    0.822902083699, 0.807732738234, 0.793171045519, 0.779139726505,
    0.765577436082, 0.752434456248, 0.739669787677, 0.727249120285,
    0.715143377413, 0.703327646455, 0.691780377035, 0.68048276891,
    0.669418297233, 0.65857233912, 0.647931876189, 0.637485254896,
    0.62722199145, 0.617132611532, 0.607208517467, 0.597441877296,
    0.587825531465, 0.578352913803, 0.569017984198, 0.559815170911,
    0.550739320877, 0.541785656682, 0.532949739145, 0.524227434628,
    0.515614886373, 0.507108489253, 0.498704867478, 0.490400854812,
    0.482193476986, 0.47407993601, 0.466057596125, 0.458123971214,
    0.450276713467, 0.442513603171, 0.434832539473, 0.427231532022,
    0.419708693379, 0.41226223212, 0.404890446548, 0.397591718955,
    0.390364510382, 0.383207355816, 0.376118859788, 0.369097692334,
    0.362142585282, 0.355252328834, 0.348425768415, 0.341661801776,
    0.334959376311, 0.328317486588, 0.321735172063, 0.31521151497,
    0.308745638367, 0.302336704338, 0.29598391232, 0.289686497571,
    0.283443729739, 0.27725491156, 0.271119377649, 0.265036493387,
    0.259005653912, 0.253026283183, 0.247097833139, 0.241219782932,
    0.235391638239, 0.229612930649, 0.223883217122, 0.218202079518,
    0.212569124201, 0.206983981709, 0.201446306496, 0.195955776745,
    0.190512094256, 0.185114984406, 0.179764196185, 0.174459502324,
    0.169200699492, 0.1639876086, 0.158820075195, 0.153697969964,
    0.148621189348, 0.143589656295, 0.138603321143, 0.133662162669,
    0.128766189309, 0.123915440582, 0.119109988745, 0.114349940703,
    0.10963544023, 0.104966670533, 0.100343857232, 0.0957672718266,
    0.0912372357329, 0.0867541250127, 0.082318375932, 0.0779304915295,
    0.0735910494266, 0.0693007111742, 0.065060233529, 0.0608704821745,
    0.056732448584, 0.05264727098, 0.0486162607163, 0.0446409359769,
    0.0407230655415, 0.0368647267386, 0.0330683839378, 0.0293369977411,
    0.0256741818288, 0.0220844372634, 0.0185735200577, 0.0151490552854,
    0.0118216532614, 0.00860719483079, 0.00553245272614, 0.00265435214565
};

/* tabulated values for 2^24 times x[i]/x[i+1],
 * used to accept for U*x[i+1]<=x[i] without any floating point operations */
static const unsigned long ktab[128] = {
    0, 12590644, 14272653, 14988939,
    15384584, 15635009, 15807561, 15933577,
    16029594, 16105155, 16166147, 16216399,
    16258508, 16294295, 16325078, 16351831,
    16375291, 16396026, 16414479, 16431002,
    16445880, 16459343, 16471578, 16482744,
    16492970, 16502368, 16511031, 16519039,
    16526459, 16533352, 16539769, 16545755,
    16551348, 16556584, 16561493, 16566101,
    16570433, 16574511, 16578353, 16581977,
    16585398, 16588629, 16591685, 16594575,
    16597311, 16599901, 16602354, 16604679,
    16606881, 16608968, 16610945, 16612818,
    16614592, 16616272, 16617861, 16619363,
    16620782, 16622121, 16623383, 16624570,
    16625685, 16626730, 16627708, 16628619,
    16629465, 16630248, 16630969, 16631628,
    16632228, 16632768, 16633248, 16633671,
    16634034, 16634340, 16634586, 16634774,
    16634903, 16634972, 16634980, 16634926,
    16634810, 16634628, 16634381, 16634066,
    16633680, 16633222, 16632688, 16632075,
    16631380, 16630598, 16629726, 16628757,
    16627686, 16626507, 16625212, 16623794,
    16622243, 16620548, 16618698, 16616679,
    16614476, 16612071, 16609444, 16606571,
    16603425, 16599973, 16596178, 16591995,
    16587369, 16582237, 16576520, 16570120,
    16562917, 16554758, 16545450, 16534739,
    16522287, 16507638, 16490152, 16468907,
    16442518, 16408804, 16364095, 16301683,
    16207738, 16047994, 15704248, 15472926
};

/* tabulated values of 2^{-24}*x[i] */
static const double wtab[128] = {
    1.62318314817e-08, 2.16291505214e-08, 2.54246305087e-08, 2.84579525938e-08,
    3.10340022482e-08, 3.33011726243e-08, 3.53439060345e-08, 3.72152672658e-08,
    3.8950989572e-08, 4.05763964764e-08, 4.21101548915e-08, 4.35664624904e-08,
    4.49563968336e-08, 4.62887864029e-08, 4.75707945735e-08, 4.88083237257e-08,
    5.00063025384e-08, 5.11688950428e-08, 5.22996558616e-08, 5.34016475624e-08,
    5.44775307871e-08, 5.55296344581e-08, 5.65600111659e-08, 5.75704813695e-08,
    5.85626690412e-08, 5.95380306862e-08, 6.04978791776e-08, 6.14434034901e-08,
    6.23756851626e-08, 6.32957121259e-08, 6.42043903937e-08, 6.51025540077e-08,
    6.59909735447e-08, 6.68703634341e-08, 6.77413882848e-08, 6.8604668381e-08,
    6.94607844804e-08, 7.03102820203e-08, 7.11536748229e-08, 7.1991448372e-08,
    7.2824062723e-08, 7.36519550992e-08, 7.44755422158e-08, 7.52952223703e-08,
    7.61113773308e-08, 7.69243740467e-08, 7.77345662086e-08, 7.85422956743e-08,
    7.93478937793e-08, 8.01516825471e-08, 8.09539758128e-08, 8.17550802699e-08,
    8.25552964535e-08, 8.33549196661e-08, 8.41542408569e-08, 8.49535474601e-08,
    8.57531242006e-08, 8.65532538723e-08, 8.73542180955e-08, 8.8156298059e-08,
    8.89597752521e-08, 8.97649321908e-08, 9.05720531451e-08, 9.138142487e-08,
    9.21933373471e-08, 9.30080845407e-08, 9.38259651738e-08, 9.46472835298e-08,
    9.54723502847e-08, 9.63014833769e-08, 9.71350089201e-08, 9.79732621669e-08,
    9.88165885297e-08, 9.96653446693e-08, 1.00519899658e-07, 1.0138063623e-07,
    1.02247952126e-07, 1.03122261554e-07, 1.04003996769e-07, 1.04893609795e-07,
    1.05791574313e-07, 1.06698387725e-07, 1.07614573423e-07, 1.08540683296e-07,
    1.09477300508e-07, 1.1042504257e-07, 1.11384564771e-07, 1.12356564007e-07,
    1.13341783071e-07, 1.14341015475e-07, 1.15355110887e-07, 1.16384981291e-07,
    1.17431607977e-07, 1.18496049514e-07, 1.19579450872e-07, 1.20683053909e-07,
    1.21808209468e-07, 1.2295639141e-07, 1.24129212952e-07, 1.25328445797e-07,
    1.26556042658e-07, 1.27814163916e-07, 1.29105209375e-07, 1.30431856341e-07,
    1.31797105598e-07, 1.3320433736e-07, 1.34657379914e-07, 1.36160594606e-07,
    1.37718982103e-07, 1.39338316679e-07, 1.41025317971e-07, 1.42787873535e-07,
    1.44635331499e-07, 1.4657889173e-07, 1.48632138436e-07, 1.50811780719e-07,
    1.53138707402e-07, 1.55639532047e-07, 1.58348931426e-07, 1.61313325908e-07,
    1.64596952856e-07, 1.68292495203e-07, 1.72541128694e-07, 1.77574279496e-07,
    1.83813550477e-07, 1.92166040885e-07, 2.05295471952e-07, 2.22600839893e-07
};

/* Binomial draw parameters */
#define SMALL_MEAN 14
#define BINV_CUTOFF 110
#define FAR_FROM_MEAN 20
#define LNFACT(x) lgamma(x+1)

/* initializes mt[N] with a seed */
void rng_init_rand( unsigned long s )
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = 
          (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void rng_init_by_array(unsigned long init_key[], int key_length)
{
    int i, j, k;
    rng_init_rand(19650218UL);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

static unsigned long get_seed_matlab( void )
{
    mxArray *plhs[1];

    mexCallMATLAB(1, plhs, 0, NULL, "rand");
    
    double r = mxGetScalar(plhs[0]);
    unsigned long seed = floor( 10000*r );
    
    mxDestroyArray(plhs[0]);

    return seed;
}

/* generates a random number on [0,0xffffffff]-interval */
static unsigned long rng_rand_int( void )
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        // if (mti == N+1)   /* if rng_init_rand() has not been called, */
        //     rng_init_rand(5489UL); /* a default initial seed is used */

        if( mti == N+1 ) 
          rng_init_rand( get_seed_matlab() );

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,1]-real-interval */
static double rng_rand_1( void )
{
    return rng_rand_int()*(1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval */
static double rng_rand_2( void )
{
    return rng_rand_int()*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double rng_rand( void )
{
    return (((double)rng_rand_int()) + 0.5)*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* Gaussian RNG from the GSL library (Ziggurat Algorithm) */
double rng_gaussian ( void )
{
    unsigned long  U, sign, i, j;
    double  x, y;
    
    while (1) {
        U = rng_rand_int();
        i = U & 0x0000007F;     /* 7 bit to choose the step */
        sign = U & 0x00000080;  /* 1 bit for the sign */
        j = U>>8;               /* 24 bit for the x-value */
        
        x = j*wtab[i];
        if (j < ktab[i])  break;
        
        if (i<127) {
            double  y0, y1;
            y0 = ytab[i];
            y1 = ytab[i+1];
            y = y1+(y0-y1)*rng_rand_2();
        } else {
            x = PARAM_R - log(1.0-rng_rand_2())/PARAM_R;
            y = exp(-PARAM_R*(x-0.5*PARAM_R))*rng_rand_2();
        }
        if (y < exp(-0.5*x*x))  break;
    }
    return  sign ? x : -x;
}

/* Gamma RNG from the GSL library */
double rng_gamma( double a,  double b )
{
  /* assume a > 0 */
  if (a < 1)
    {
      double u = rng_rand();
      return rng_gamma (1.0 + a, b) * pow (u, 1.0 / a);
    }

  {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1)
      {
        do
          {
            x = rng_gaussian();
            v = 1.0 + c * x;
          }
        while (v <= 0);

        v = v * v * v;
        u = rng_rand();

        if (u < 1 - 0.0331 * x * x * x * x) 
          break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
          break;
      }
    
    return b * d * v;
  }
}

/* Exponential RNG from the GSL library */
double rng_exp( double lambda )
{
  double u = rng_rand_2();

  return -lambda * log1p(-u);
}

double rng_chi2( double v )
{
    return 2 * rng_gamma(v/2, 1.0);
}

double rng_t( double k )
{
    double z = rng_gaussian();
    double c = rng_chi2(k);
    
    return z*sqrt(k/c);
}

double rng_beta( double a,  double b )
{
    double x = rng_gamma(a,1);
    double y = rng_gamma(b,1);
    return x/(x+y);
}

/* efficient power computation */
double int_pow( double base, int exp )
{
    double result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= (double) base;
        exp >>= 1;
        base *= base;
    }
    
    return result;
}

inline static double Stirling(double y1)
{
  double y2 = y1 * y1;
  double s =
    (13860.0 -
     (462.0 - (132.0 - (99.0 - 140.0 / y2) / y2) / y2) / y2) / y1 / 166320.0;
  return s;
}

/* gsl_ran_binomial i.e the default method in GSL */
int rng_binomial( double p,  int n )
{
  int ix;                       /* return value */
  int flipped = 0;
  double q, s, np;

  if (n == 0)
    return 0;

  if (p > 0.5)
    {
      p = 1.0 - p;              /* work with small p */
      flipped = 1;
    }

  q = 1 - p;
  s = p / q;
  np = n * p;

  /* Inverse cdf logic for small mean (BINV in K+S) */

  if (np < SMALL_MEAN)
    {
      double f0 = int_pow(q, n);   /* f(x), starting with x=0 */

      while (1)
        {
          /* This while(1) loop will almost certainly only loop once; but
           * if u=1 to within a few epsilons of machine precision, then it
           * is possible for roundoff to prevent the main loop over ix to
           * achieve its proper value.  following the ranlib implementation,
           * we introduce a check for that situation, and when it occurs,
           * we just try again.
           */

          double f = f0;
          double u = rng_rand();

          for (ix = 0; ix <= BINV_CUTOFF; ++ix)
            {
              if (u < f)
                goto Finish;
              u -= f;
              /* Use recursion f(x+1) = f(x)*[(n-x)/(x+1)]*[p/(1-p)] */
              f *= s * (n - ix) / (ix + 1);
            }

          /* It should be the case that the 'goto Finish' was encountered
           * before this point was ever reached.  But if we have reached
           * this point, then roundoff has prevented u from decreasing
           * all the way to zero.  This can happen only if the initial u
           * was very nearly equal to 1, which is a rare situation.  In
           * that rare situation, we just try again.
           *
           * Note, following the ranlib implementation, we loop ix only to
           * a hardcoded value of SMALL_MEAN_LARGE_N=110; we could have
           * looped to n, and 99.99...% of the time it won't matter.  This
           * choice, I think is a little more robust against the rare
           * roundoff error.  If n>LARGE_N, then it is technically
           * possible for ix>LARGE_N, but it is astronomically rare, and
           * if ix is that large, it is more likely due to roundoff than
           * probability, so better to nip it at LARGE_N than to take a
           * chance that roundoff will somehow conspire to produce an even
           * larger (and more improbable) ix.  If n<LARGE_N, then once
           * ix=n, f=0, and the loop will continue until ix=LARGE_N.
           */
        }
    }
  else
    {
      /* For n >= SMALL_MEAN, we invoke the BTPE algorithm */

      int k;

      double ffm = np + p;      /* ffm = n*p+p             */
      int m = (int) ffm;        /* m = int floor[n*p+p]    */
      double fm = m;            /* fm = double m;          */
      double xm = fm + 0.5;     /* xm = half integer mean (tip of triangle)  */
      double npq = np * q;      /* npq = n*p*q            */

      /* Compute cumulative area of tri, para, exp tails */

      /* p1: radius of triangle region; since height=1, also: area of region */
      /* p2: p1 + area of parallelogram region */
      /* p3: p2 + area of left tail */
      /* p4: p3 + area of right tail */
      /* pi/p4: probability of i'th area (i=1,2,3,4) */

      /* Note: magic numbers 2.195, 4.6, 0.134, 20.5, 15.3 */
      /* These magic numbers are not adjustable...at least not easily! */

      double p1 = floor (2.195 * sqrt (npq) - 4.6 * q) + 0.5;

      /* xl, xr: left and right edges of triangle */
      double xl = xm - p1;
      double xr = xm + p1;

      /* Parameter of exponential tails */
      /* Left tail:  t(x) = c*exp(-lambda_l*[xl - (x+0.5)]) */
      /* Right tail: t(x) = c*exp(-lambda_r*[(x+0.5) - xr]) */

      double c = 0.134 + 20.5 / (15.3 + fm);
      double p2 = p1 * (1.0 + c + c);

      double al = (ffm - xl) / (ffm - xl * p);
      double lambda_l = al * (1.0 + 0.5 * al);
      double ar = (xr - ffm) / (xr * q);
      double lambda_r = ar * (1.0 + 0.5 * ar);
      double p3 = p2 + c / lambda_l;
      double p4 = p3 + c / lambda_r;

      double var, accept;
      double u, v;              /* random variates */

    TryAgain:

      /* generate random variates, u specifies which region: Tri, Par, Tail */
      u = rng_rand () * p4;
      v = rng_rand ();

      if (u <= p1)
        {
          /* Triangular region */
          ix = (int) (xm - p1 * v + u);
          goto Finish;
        }
      else if (u <= p2)
        {
          /* Parallelogram region */
          double x = xl + (u - p1) / c;
          v = v * c + 1.0 - fabs (x - xm) / p1;
          if (v > 1.0 || v <= 0.0)
            goto TryAgain;
          ix = (int) x;
        }
      else if (u <= p3)
        {
          /* Left tail */
          ix = (int) (xl + log (v) / lambda_l);
          if (ix < 0)
            goto TryAgain;
          v *= ((u - p2) * lambda_l);
        }
      else
        {
          /* Right tail */
          ix = (int) (xr - log (v) / lambda_r);
          if (ix > (double) n)
            goto TryAgain;
          v *= ((u - p3) * lambda_r);
        }

      /* At this point, the goal is to test whether v <= f(x)/f(m) 
       *
       *  v <= f(x)/f(m) = (m!(n-m)! / (x!(n-x)!)) * (p/q)^{x-m}
       *
       */

      /* Here is a direct test using logarithms.  It is a little
       * slower than the various "squeezing" computations below, but
       * if things are working, it should give exactly the same answer
       * (given the same random number seed).  */

#ifdef DIRECT
      var = log (v);

      accept =
        LNFACT (m) + LNFACT (n - m) - LNFACT (ix) - LNFACT (n - ix)
        + (ix - m) * log (p / q);

#else /* SQUEEZE METHOD */

      /* More efficient determination of whether v < f(x)/f(M) */

      k = abs (ix - m);

      if (k <= FAR_FROM_MEAN)
        {
          /* 
           * If ix near m (ie, |ix-m|<FAR_FROM_MEAN), then do
           * explicit evaluation using recursion relation for f(x)
           */
          double g = (n + 1) * s;
          double f = 1.0;

          var = v;

          if (m < ix)
            {
              int i;
              for (i = m + 1; i <= ix; i++)
                {
                  f *= (g / i - s);
                }
            }
          else if (m > ix)
            {
              int i;
              for (i = ix + 1; i <= m; i++)
                {
                  f /= (g / i - s);
                }
            }

          accept = f;
        }
      else
        {
          /* If ix is far from the mean m: k=ABS(ix-m) large */

          var = log (v);

          if (k < npq / 2 - 1)
            {
              /* "Squeeze" using upper and lower bounds on
               * log(f(x)) The squeeze condition was derived
               * under the condition k < npq/2-1 */
              double amaxp =
                k / npq * ((k * (k / 3.0 + 0.625) + (1.0 / 6.0)) / npq + 0.5);
              double ynorm = -(k * k / (2.0 * npq));
              if (var < ynorm - amaxp)
                goto Finish;
              if (var > ynorm + amaxp)
                goto TryAgain;
            }

          /* Now, again: do the test log(v) vs. log f(x)/f(M) */

#if USE_EXACT
          /* This is equivalent to the above, but is a little (~20%) slower */
          /* There are five log's vs three above, maybe that's it? */

          accept = LNFACT (m) + LNFACT (n - m)
            - LNFACT (ix) - LNFACT (n - ix) + (ix - m) * log (p / q);

#else /* USE STIRLING */
          /* The "#define Stirling" above corresponds to the first five
           * terms in asymptoic formula for
           * log Gamma (y) - (y-0.5)log(y) + y - 0.5 log(2*pi);
           * See Abramowitz and Stegun, eq 6.1.40
           */

          /* Note below: two Stirling's are added, and two are
           * subtracted.  In both K+S, and in the ranlib
           * implementation, all four are added.  I (jt) believe that
           * is a mistake -- this has been confirmed by personal
           * correspondence w/ Dr. Kachitvichyanukul.  Note, however,
           * the corrections are so small, that I couldn't find an
           * example where it made a difference that could be
           * observed, let alone tested.  In fact, define'ing Stirling
           * to be zero gave identical results!!  In practice, alv is
           * O(1), ranging 0 to -10 or so, while the Stirling
           * correction is typically O(10^{-5}) ...setting the
           * correction to zero gives about a 2% performance boost;
           * might as well keep it just to be pendantic.  */

          {
            double x1 = ix + 1.0;
            double w1 = n - ix + 1.0;
            double f1 = fm + 1.0;
            double z1 = n + 1.0 - fm;

            accept = xm * log (f1 / x1) + (n - m + 0.5) * log (z1 / w1)
              + (ix - m) * log (w1 * p / (x1 * q))
              + Stirling (f1) + Stirling (z1) - Stirling (x1) - Stirling (w1);
          }
#endif
#endif
        }


      if (var <= accept)
        {
          goto Finish;
        }
      else
        {
          goto TryAgain;
        }
    }

Finish:

  return (flipped) ? (n - ix) : (unsigned int)ix;
}

/* gsl_ran_poisson ( same routine as in Matlab ) */
int rng_poisson( double mu )
{
  double emu;
  double prod = 1.0;
  int k = 0;

  while (mu > 10)
    {
      int m = mu * (7.0 / 8.0);

      double X = rng_gamma( m, 1);

      if (X >= mu)
        {
          return k + rng_binomial ( mu / X, m - 1 );
        }
      else
        {
          k += m;
          mu -= X; 
        }
    }

  /* This following method works well when mu is small */

  emu = exp (-mu);

  do
    {
      prod *= rng_rand();
      k++;
    }
  while (prod > emu);

  return k - 1;

}

int rng_n_binomial( double p, double n )
{
  double X = rng_gamma(n, 1.0);
  int k = rng_poisson( X*(1-p)/p );
  return k ;
}

double rng_chi2_pdf ( double x, double nu)
{
  if (x < 0)
    {
      return 0 ;
    }
  else
    {
      if(nu == 2.0)
        {
          return exp(-x/2.0) / 2.0;
        }
      else
        {
          double p;
          
          p = exp ((nu / 2 - 1) * log (x/2) - x/2 - lgamma(nu/2)) / 2;
          return p;
        }
    }
}       
        
/* Homer's Method */
double poly_eval(double *coeffs, int s, double x)
{
  int i;
  double res = 0.0;
 
  for(i=s-1; i >= 0; i--)
  {
    res = res * x + coeffs[i];
  }
  return res;
}