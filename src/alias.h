#ifndef ALIAS
#define ALIAS

// For internal C use
void draw_discrete(int n, double *p, int n_draws, int *draws);
void alias_tables(int n, double *p, int *Alias, double *Prob);
void draw_discrete_from_alias_tables(int n, int *Alias, double *Prob, int n_draws, int *draws);

#endif /* ALIAS */
