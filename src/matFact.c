#include <stdio.h>
#include <stdlib.h>

#define RAND01 ((double)random() / (double)RAND_MAX)
#define DEBUG  1

// GLOBAL VARIABLES
double **A, **B, **L, **R;

int parse_int(FILE *fp) {
  int value;
  if (1 != fscanf(fp, "%d", &value)) {
    fprintf(stderr, "Unable to parse unsigned int.\n");
    exit(1);
  }
  return value;
}

double parse_double(FILE *fp) {
  double value;
  if (1 != fscanf(fp, "%lf", &value)) {
    fprintf(stderr, "Unable to parse double.\n");
    exit(1);
  }
  return value;
}

double **new_matrix(int l, int c) {
  int i, j;

  double **matrix = (double **) malloc(sizeof(double *) * l);
  if (NULL == matrix)
    return NULL;

  for (i = 0; i < l; i++) {
    matrix[i] = (double *) malloc(sizeof(double) * c);
    for (j = 0; j < c; j++)
      matrix[i][j] = 0;
  }

  return matrix;
}

void delete_matrix(double **matrix, int l) {
  int i;

  for (i = 0; i < l; i++)
    free(matrix[i]);
  free(matrix);
}

void copy_matrix(double **from, double **to, int l, int c) {
  int i, j;
  
  for (i = 0; i < l; i++)
    for (j = 0; j < c; j++)
      from[i][j] = to[i][j];
}

void print_matrix(double **matrix, int l, int c) {
  int i, j;

  for (i = 0; i < l; i++) {
    for (j= 0; j < c; j++)
      printf("%1.6lf ", matrix[i][j]);
    printf("\n");
  }
}

void random_fill_LR(int nU, int nI, int nF) {
  int i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L[i][j] = RAND01 / (double) nF;
  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R[i][j] = RAND01 / (double) nF;
}

int main(int argc, char **argv) {
  FILE *fp;
  char *fname;
  double alpha;
  int iters, nz_lines;
  int n_users, n_items, n_feats;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <instance>\n", argv[0]);
    exit(1);
  }

  fname = argv[1];
  if (NULL == (fp = fopen(fname, "r"))) {
    fprintf(stderr, "Unable to open file: \'%s\'\n", fname);
    exit(1);
  }

  iters = parse_int(fp);
  alpha = parse_double(fp);
  n_feats = parse_int(fp);
  n_users = parse_int(fp); 
  n_items = parse_int(fp);
  nz_lines = parse_int(fp);
  
  if (DEBUG) {
    printf("%d %1.6lf %d %d %d %d\n", iters, alpha, n_feats,
        n_users, n_items, nz_lines);
  }

  A = new_matrix(n_users, n_items);
  while (nz_lines--) {
    int i = parse_int(fp), j = parse_int(fp);
    A[i][j] = parse_double(fp);
  }
  
  if (0 != fclose(fp)) {
    fprintf(stderr, "Unable to flush file stream.\n");
    exit(1);
  }

  if (DEBUG) {
    printf("Initial matrix A\n");
    print_matrix(A, n_users, n_items);
  }

  L = new_matrix(n_users, n_feats);
  R = new_matrix(n_feats, n_items);
  B = new_matrix(n_users, n_items);
  random_fill_LR(n_users, n_items, n_feats);

  if (DEBUG) {
    printf("Initial matrix L\n");
    print_matrix(L, n_users, n_feats);

    printf("Initial matrix R\n");
    print_matrix(R, n_feats, n_items);
  }

  delete_matrix(B, n_users);
  delete_matrix(R, n_feats);
  delete_matrix(L, n_users);
  delete_matrix(A, n_users);

  return 0;
}
