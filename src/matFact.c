#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

// global variables ----------------------------------------
double **A, **B;
double **Lt, **L;
double **Rt, **R;

// parsing helpers -----------------------------------------
unsigned int parse_uint(FILE *fp) {
  unsigned int value;
  if (1 != fscanf(fp, "%u", &value)) {
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

// matrix helpers ------------------------------------------
double **new_matrix(unsigned int l, unsigned int c) {
  size_t i, j;

  double **m = (double **) malloc(sizeof(double *) * l);
  if (NULL == m)
    return NULL;

  for (i = 0; i < l; i++) {
    m[i] = (double *) malloc(sizeof(double) * c);
    for (j = 0; j < c; j++) {
      m[i][j] = '\x00';
    }
  }

  return m;
}

void delete_matrix(double **m, unsigned int l) {
  size_t i;

  for (i = 0; i < l; i++)
    free(m[i]);
  free(m);
  return;
}

void print_matrix(double **m, unsigned int l, unsigned int c) {
  size_t i, j;

  for (i = 0; i < l; i++) {
    for (j= 0; j < c; j++) {
      printf("%1.6lf ", m[i][j]);
    }
    printf("\n");
  }
  return;
}

// matrix operations ---------------------------------------
void random_fill_LR(unsigned int nU, unsigned int nI,
    unsigned int nF) {
  size_t i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L[i][j] = RAND01 / (double) nF;

  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R[i][j] = RAND01 / (double) nF;
  return;
}

void matrix_mult_LR(unsigned int nU, unsigned int nI,
    unsigned int nF) {
  size_t i, j, k;

  for (i = 0; i < nU; i++) {
    for (j = 0; j < nI; j++) {
      double sum = 0;
      for (k = 0; k < nF; k++) {
        sum += L[i][k] * R[k][j];
      }
      B[i][j] = sum;
    }
  }
  return;
}

void matrix_fact_B(unsigned int nU, unsigned int nI,
    unsigned int nF, unsigned int n, double a) {
  size_t i, j, k;
  double **tmp;

  do {
    tmp = L; L = Lt; Lt = tmp;
    tmp = R; R = Rt; Rt = tmp;

    for (i = 0; i < nU; i++) {
      for (k = 0; k < nF; k++) {
        double sum = 0;
        for (j = 0; j < nI; j++) {
          if (A[i][j])
            sum += 2*(A[i][j] - B[i][j])*(-Rt[k][j]);
        }
        L[i][k] = Lt[i][k] - (a * sum);
      }
    }

    for (k = 0; k < nF; k++) {
      for (j = 0; j < nI; j++) {
        double sum = 0;
        for (i = 0; i < nU; i++) {
          if (A[i][j])
            sum += 2*(A[i][j] - B[i][j])*(-Lt[i][k]);
        }
        R[k][j] = Rt[k][j] - (a * sum);
      }
    }

    matrix_mult_LR(nU, nI, nF);
  } while(--n);
  return;
}

void recommend(unsigned int l, unsigned int c) {
  size_t i, j;

  for (i = 0; i < l; i++) {
    size_t item = 0;
    double max = -1.00;
    for (j = 0; j < c; j++) {
      if ((B[i][j] > max) && !A[i][j]) {
        max = B[i][j];
        item = j;
      }
    }
    printf("%lu\n", item);
  }
  return;
}

// main ----------------------------------------------------
int main(int argc, char **argv) {
  FILE *fp;
  double alpha;
  size_t i, j;
  unsigned int N, lines;
  unsigned int numU, numI, numF;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <instance>\n", argv[0]);
    return 1;
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Unable to open file: \'%s\'\n", argv[1]);
    return 1;
  }

  N = parse_uint(fp);
  alpha = parse_double(fp);
  numF = parse_uint(fp);
  numU = parse_uint(fp); numI = parse_uint(fp); lines = parse_uint(fp);

  A = new_matrix(numU, numI);
  do {
    i = parse_uint(fp);
    j = parse_uint(fp);
    A[i][j] = parse_double(fp);
  } while(--lines);

  if (fclose(fp) != 0) {
    fprintf(stderr, "Unable to flush file stream.\n");
    return 1;
  }

  L  = new_matrix(numU, numF);
  Lt = new_matrix(numU, numF);
  R  = new_matrix(numF, numI);
  Rt = new_matrix(numF, numI);
  B  = new_matrix(numU, numI);

  random_fill_LR(numU, numI, numF);
  matrix_mult_LR(numU, numI, numF);
  matrix_fact_B(numU, numI, numF, N, alpha);
  print_matrix(B, numU, numI);
  recommend(numU, numI);

  delete_matrix(B, numU);
  delete_matrix(Rt, numF); 
  delete_matrix(R, numF);
  delete_matrix(Lt, numU);
  delete_matrix(L, numU);
  delete_matrix(A, numU);
  return 0;
}
