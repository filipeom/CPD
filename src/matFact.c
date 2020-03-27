#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef unsigned int uint;

struct csr {
  uint *row;
  uint *col;
  double *val;
} __attribute__((aligned(64)));

/* globals */
static char *argv0 = NULL;
static struct csr *A = NULL;

static double *B  = NULL;
static double *Lt = NULL;
static double *L  = NULL;
static double *Rt = NULL;
static double *R  = NULL;

/* general helpers */
void
die(const char *err_str, ...)
{
  va_list ap;

  va_start(ap, err_str);
  vfprintf(stderr, err_str, ap);
  va_end(ap);
  exit(1);
}

void
usage(void)
{
  die("usage: %s INSTANCE\n", argv0);
}

/* parsing helpers */
uint
parse_uint(FILE *fp)
{
  uint value;

  if (1 != fscanf(fp, "%u", &value))
    die("unable to parse integer.\n");
  return value;
}

double
parse_double(FILE *fp)
{
  double value;

  if (1 != fscanf(fp, "%lf", &value))
    die("unable to parse double.\n");
  return value;
}

/* matrix helpers */
void
matrix_init(double *matrix[], uint l, uint c)
{
  *matrix = (double *) malloc(sizeof(double) * l * c);
  if (NULL == *matrix)
    die("unable to allocate memory for matrix.\n");

  memset(*matrix, '\x00', sizeof(double) * l * c);
}

void
matrix_destroy(double *matrix)
{
  if (NULL != matrix)
    free(matrix);
}

void
print_matrix(const double *matrix, uint l, uint c)
{
  size_t i, j;

  for (i = 0; i < l; ++i) {
    for (j= 0; j < c; ++j) {
      printf("%1.6lf ", matrix[i*c + j]);
    }
    printf("\n");
  }
}

void
csr_matrix_init(struct csr **matrix, uint nnz, uint nU)
{
  *matrix = (struct csr *) malloc(sizeof(struct csr));
  if (NULL == *matrix)
    die("unable to allocate memory for sparse matrix.\n");

  (*matrix)->row = (uint *) malloc(sizeof(uint) * (nU + 1));
  (*matrix)->col = (uint *) malloc(sizeof(uint) * nnz);
  (*matrix)->val = (double *) malloc(sizeof(double) * nnz);

  if (NULL == (*matrix)->row || NULL == (*matrix)->col ||
      NULL == (*matrix)->val)
    die("unable to allocate memory for sparse matrix.\n");

  memset((*matrix)->row, '\x00', sizeof(uint) * (nU + 1));
}

void
csr_matrix_destroy(struct csr *matrix)
{
  if (NULL != matrix->row && NULL != matrix->col &&
      NULL != matrix->val && NULL != matrix) {
    free(matrix->row);
    free(matrix->col);
    free(matrix->val);
    free(matrix);
  }
}

/* matrix operations */
void
random_fill_LR(uint nU, uint nI, uint nF)
{
  size_t i, j;

  srandom(0);

  for (i = 0; i < nU; ++i)
    for (j = 0; j < nF; ++j)
      L[i*nF + j] = RAND01 / (double) nF;

  for (i = 0; i < nF; ++i)
    for (j = 0; j < nI; ++j)
      R[j*nF + i] = RAND01 / (double) nF;
}

void
run(uint n, double alpha, uint nU, uint nI, uint nF)
{
  size_t i, j, k;
  size_t jx, item;
  double tmp;
  double *restrict m1;
  double *restrict m2;
  double *restrict mres;

  /* matrix factorization */
  for (size_t it = n; it--; ) {
    memcpy(Rt, R, sizeof(double) * nI * nF);
    memcpy(Lt, L, sizeof(double) * nU * nF);

    for (i = 0; i < nU; ++i) {
      for (jx = A->row[i]; jx < A->row[i+1]; ++jx) {
        j = A->col[jx];
        m1 = &L[i*nF];
        m2 = &R[j*nF];
        tmp = 0;
        for (k = 0; k < nF; ++k) {
          tmp += Lt[i*nF+k] * Rt[j*nF + k];
        }
        tmp = A->val[jx] - tmp;
        for (k = 0; k < nF; ++k) {
          m1[k] += alpha * 2 * tmp * Rt[j*nF + k];
          m2[k] += alpha * 2 * tmp * Lt[i*nF + k];
        }
      }
    }
  }

  /* matrix multiplication */
  for (i = 0, mres = &B[i*nI], m1 = &L[i*nF]; i < nU;
      ++i, mres += nI, m1 += nF) {
    for (j = 0, m2 = &R[j*nF]; j < nI; ++j, m2 += nF) {
      tmp = 0;
      for (k = 0; k < nF; ++k) {
        tmp += m1[k] * m2[k];
      }
      mres[j] = tmp;
    }
  }

  /* FIXME: output */
  for (i = 0; i < nU; ++i)
    for (jx = A->row[i]; jx < A->row[i+1]; ++jx)
      B[i*nI + A->col[jx]] = 0;
  for (i = 0; i < nU; ++i) {
    item = 0; tmp = -1.0;
    for (j = 0; j < nI; ++j) {
      if (B[i*nI + j] > tmp) {
        tmp = B[i*nI + j];
        item = j;
      }
    }
    printf("%lu\n", item);
  }
}

/* main */
int
main(int argc, char *argv[])
{
  FILE *fp;
  size_t i, ij;

  argv0 = argv[0];
  if (2 != argc)
    usage();

  if (NULL == (fp = fopen(argv[1], "r")))
    die("unable to open file: \'%s\'\n", argv[1]);

  uint N   = parse_uint(fp);
  double a = parse_double(fp);
  uint nF  = parse_uint(fp);
  uint nU  = parse_uint(fp);
  uint nI  = parse_uint(fp);
  uint nnz = parse_uint(fp);

  csr_matrix_init(&A, nnz, nU);
  for (ij = 0; ij < nnz; ++ij) {
    i = parse_uint(fp);
    A->row[i+1] += 1;
    A->col[ij] = parse_uint(fp);
    A->val[ij] = parse_double(fp);
  }
  for (i = 1; i <= nU; ++i)
    A->row[i] += A->row[i-1];

  if (0 != fclose(fp))
    die("unable to flush file stream.\n");

  matrix_init(&L, nU, nF);
  matrix_init(&Lt, nU, nF);
  matrix_init(&R, nI, nF);
  matrix_init(&Rt, nI, nF);
  matrix_init(&B, nU, nI);

  random_fill_LR(nU, nI, nF);

  run(N, a, nU, nI, nF);

  matrix_destroy(B);
  matrix_destroy(Rt);
  matrix_destroy(R);
  matrix_destroy(Lt);
  matrix_destroy(L);
  csr_matrix_destroy(A);

  return 0;
}
