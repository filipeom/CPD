#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

/* globals */
static double **A  = NULL;
static double **B  = NULL;
static double **Lt = NULL;
static double **L  = NULL;
static double **Rt = NULL;
static double **R  = NULL;

static char *argv0;

/* general helpers */
void
die(const char *err_str, ...) {
  va_list ap;

  va_start(ap, err_str);
  vfprintf(stderr, err_str, ap);
  va_end(ap);
  exit(1);
}

void
usage(void) {
  die("usage: %s INSTANCE\n", argv0);
}

/* parsing helpers */
u_int32_t
parse_uint(FILE *fp) {
  u_int32_t value;

  if (1 != fscanf(fp, "%u", &value))
    die("unable to parse integer.\n");
  return value;
}

double 
parse_double(FILE *fp) {
  double value;

  if (1 != fscanf(fp, "%lf", &value))
    die("unable to parse double.\n");
  return value;
}

/* matrix helpers */
double** 
new_matrix(u_int32_t l, u_int32_t c) {
  size_t i, j;

  double **m = (double **) malloc(sizeof(double *) * l);
  if (NULL == m)
    die("unable to allocate memory for matrix.\n");

  for (i = 0; i < l; i++) {
    m[i] = (double *) malloc(sizeof(double) * c);
    for (j = 0; j < c; j++) {
      m[i][j] = '\x00';
    }
  }

  return m;
}

void 
delete_matrix(double **m, u_int32_t l) {
  size_t i;

  for (i = 0; i < l; i++)
    free(m[i]);
  free(m);
}

void 
print_matrix(double **m, u_int32_t l, u_int32_t c) {
  size_t i, j;

  for (i = 0; i < l; i++) {
    for (j= 0; j < c; j++) {
      printf("%1.6lf ", m[i][j]);
    }
    printf("\n");
  }
}

/* matrix operations */
void
random_fill_LR(u_int32_t nU, u_int32_t nI, u_int32_t nF) {
  size_t i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L[i][j] = RAND01 / (double) nF;

  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R[i][j] = RAND01 / (double) nF;
}

void
matrix_mult_LR(u_int32_t nU, u_int32_t nI, u_int32_t nF) {
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
}

void
matrix_fact_B(u_int32_t n, double a, u_int32_t nU, u_int32_t nI,
    u_int32_t nF) {
  size_t i, j, k;
  double **tmp;

  do {
    tmp = L; L = Lt; Lt = tmp;
    tmp = R; R = Rt; Rt = tmp;

    for (i = 0; i < nU; i++) {
      for (k = 0; k < nF; k++) {
        double sum = 0;
        for (j = 0; j < nI; j++) {
          if (A[i][j]) sum += 2*(A[i][j] - B[i][j])*(-Rt[k][j]);
        }
        L[i][k] = Lt[i][k] - (a * sum);
      }
    }

    for (k = 0; k < nF; k++) {
      for (j = 0; j < nI; j++) {
        double sum = 0;
        for (i = 0; i < nU; i++) {
          if (A[i][j]) sum += 2*(A[i][j] - B[i][j])*(-Lt[i][k]);
        }
        R[k][j] = Rt[k][j] - (a * sum);
      }
    }

    matrix_mult_LR(nU, nI, nF);
  } while(--n);
}

void
recommend(u_int32_t l, u_int32_t c) {
  double max;
  size_t i, j, item;

  for (i = 0; i < l; i++) {
    item = 0;
    max = -1.00;
    for (j = 0; j < c; j++) {
      if ((B[i][j] > max) && !A[i][j]) {
        max = B[i][j];
        item = j;
      }
    }
    printf("%lu\n", item);
  }
}

/* main */
int
main(int argc, char **argv) {
  FILE *fp;
  double alpha;
  size_t i, j;
  u_int32_t N, lines;
  u_int32_t numU, numI, numF;

  argv0 = argv[0];
  if (argc != 2)
    usage();

  if (NULL == (fp = fopen(argv[1], "r")))
    die("unable to open file: \'%s\'\n", argv[1]);

  N = parse_uint(fp);
  alpha = parse_double(fp);
  numF = parse_uint(fp);
  numU = parse_uint(fp);
  numI = parse_uint(fp);
  lines = parse_uint(fp);

  A = new_matrix(numU, numI);
  do {
    i = parse_uint(fp);
    j = parse_uint(fp);
    A[i][j] = parse_double(fp);
  } while(--lines);

  if (0 != fclose(fp))
    die("unable to flush file stream.\n");

  L  = new_matrix(numU, numF);
  Lt = new_matrix(numU, numF);
  R  = new_matrix(numF, numI);
  Rt = new_matrix(numF, numI);
  B  = new_matrix(numU, numI);

  random_fill_LR(numU, numI, numF);
  matrix_mult_LR(numU, numI, numF);
  matrix_fact_B(N, alpha, numU, numI, numF);
  recommend(numU, numI);

  delete_matrix(B, numU);
  delete_matrix(Rt, numF); 
  delete_matrix(R, numF);
  delete_matrix(Lt, numU);
  delete_matrix(L, numU);
  delete_matrix(A, numU);
  return 0;
}
