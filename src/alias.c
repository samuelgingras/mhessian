#include <math.h>
#include "alias.h"
#include "RNG.h"
#include "Phi.h"

// Vose's Alias Method to draw a discrete random variable
#define max_n 100

void draw_discrete_from_alias_tables(int n, int *Alias, double *Prob, int n_draws, int *draws)
{
  int i, k;
  for (i=0; i<n_draws; i++) {
    k = floor(n * rng_rand());
    draws[i] = (rng_rand() < Prob[k]) ? k : Alias[k];
  }
}

void alias_tables(int n, double *p, int *Alias, double *Prob)
{
  int k, l, g;
  double sum = 0.0;
  double P[max_n]; // Normalized probabilities times n

  // 2. Create two worklists, Small and Large
  int Large[max_n]; int n_Large = 0;
  int Small[max_n]; int n_Small = 0;

  // 3. Multiply each probability by n
  for (k=0; k<n; k++)
    sum += p[k];
  for (k=0; k<n; k++)
    P[k] = p[k] * n / sum;

  // 4. For each scaled probability P[k]:
  //    (a) if P[k] < 1, add i to Small;
  //    (b) otherwise add i to Large
  for (k=0; k<n; k++)
    if (P[k] < 1) Small[n_Small++] = k; else Large[n_Large++] = k;

  // 5. While Small and Large are not empty:
  for (; n_Large > 0 && n_Small > 0; ) {
    // (a) Remove first element from Small, call it l
    l = Small[--n_Small];
    // (b) Remove first element from Large, call it g
    g = Large[--n_Large];
    // (c) and (d) Set prob[l] = p[l], alias[l] = g
    Prob[l] = P[l];
    Alias[l] = g;
    // (e) Set p_g = P[g] + P[l] - 1
    P[g] = P[g] + P[l] - 1.0;
    // (f) and (g) If p_g < 1, add g to small; otherwise add g to Large
    if (P[g] < 1) Small[n_Small++] = g; else Large[n_Large++] = g;
  }

  // 6. While Large is not empty: remove first element from Large, call it g, set P[g] = 1
  for (; n_Large > 0; ) {
    P[Large[--n_Large]] = 1.0;
    Alias[Large[n_Large]] = Large[n_Large];
  }

  // 7. While Small is not empty: remove first element from Small, call it l, set P[l] = 1
  for (; n_Small > 0; ) {
    P[Small[--n_Small]] = 1.0;
    Alias[Small[n_Small]] = Small[n_Small];
  }
}

void draw_discrete(int n, double *p, int n_draws, int *draws)
{
  int Alias[max_n];
  double Prob[max_n];
  alias_tables(n, p, Alias, Prob);
  draw_discrete_from_alias_tables(n, Alias, Prob, n_draws, draws);
}