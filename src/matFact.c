#include <stdio.h>
#include <stdlib.h>

#define RAND01 ((double)random() / (double)RAND_MAX)
#define DEBUG  1

/* global variables */
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
    for (j = 0; j < c; j++) {
      matrix[i][j] = 0;  
    }
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
    for (j= 0; j < c; j++) {
      printf("%1.6lf ", matrix[i][j]);
    }
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

void matrix_multiply(int nU, int nI, int nF) {
  int i, j, k;

  for (i = 0; i < nU; i++) {
    for (j = 0; j < nI; j++) {
      double sum = 0;
      for (k = 0; k < nF; k++) {
        sum += L[i][k] * R[k][j];
      }
      B[i][j] = sum;
    }
  }
}

int main(int argc, char **argv) {
  FILE *fp;
  char *fname;
  double alpha;
  int iters, nz_lines;
  int n_users, n_items, n_feats;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <instance>\n", argv[0]);
    return 1;
  }

  fname = argv[1];
  if (NULL == (fp = fopen(fname, "r"))) {
    fprintf(stderr, "Unable to open file: \'%s\'\n", fname);
    return 1;
  }

  iters = parse_int(fp);      /* iteration count */
  alpha = parse_double(fp);   /* alpha constant */
  n_feats = parse_int(fp);    /* number of latent features to consider */
  n_users = parse_int(fp);    /* number of rows */
  n_items = parse_int(fp);    /* number of columns */
  nz_lines = parse_int(fp);   /* number of non-zero elements in A */

  /* input matrix */
  A = new_matrix(n_users, n_items);
  /* parse non-zero values into input matrix */
  while (nz_lines--)
    A[parse_int(fp)][parse_int(fp)] = parse_double(fp);

  if (0 != fclose(fp)) {
    fprintf(stderr, "Unable to flush file stream.\n");
    return 1;
  }

  L = new_matrix(n_users, n_feats);
  R = new_matrix(n_feats, n_items);
  random_fill_LR(n_users, n_items, n_feats);

  B = new_matrix(n_users, n_items);
  matrix_multiply(n_users, n_items, n_feats);
  print_matrix(B, n_users, n_items);

  /* delete matrices */
  delete_matrix(B, n_users);
  delete_matrix(R, n_feats);
  delete_matrix(L, n_users);
  delete_matrix(A, n_users);

  return 0;
}
