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
double*
new_matrix(const char *mn, uint l, uint c)
{
  double *m = (double *) malloc(sizeof(double) * l * c);
  if (NULL == m)
    die("unable to allocate memory for matrix \'%s\'.\n", mn);

  memset(m, '\x00', sizeof(double)*l*c);

  return m;
}

void
delete_matrix(double *m)
{
  if (NULL != m)
    free(m);
}

void
print_matrix(const double *m, uint l, uint c)
{
  size_t i, j;

  for (i = 0; i < l; ++i) {
    for (j= 0; j < c; ++j) {
      printf("%1.6lf ", m[i*c + j]);
    }
    printf("\n");
  }
}

struct csr*
new_csr_matrix(const char *mn, uint num_nz)
{
  struct csr *m;

  m = (struct csr *) malloc(sizeof(struct csr));
  if (NULL == m)
    die("unable to allocate memory for sparse matrix \'%s\'.\n", mn);

  m->row = (uint *) malloc(sizeof(uint) * num_nz);
  m->col = (uint *) malloc(sizeof(uint) * num_nz);
  m->val = (double *) malloc(sizeof(double) * num_nz);

  if (NULL == m->row || NULL == m->col || NULL == m->val)
    die("unable to allocate memory for sparse matrix \'%s\'.\n", mn);

  return m;
}

void
delete_csr_matrix(struct csr *m)
{
  if (m->row && m->row && m->row && m) {
    free(m->row);
    free(m->col);
    free(m->val);
    free(m);
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
matrix_mult_LR(uint nU, uint nI, uint nF)
{
  size_t i, j, k;
  double tmp;
  double *restrict m1;
  double *restrict m2;
  double *restrict mres;

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
}

void
run(uint n, double alpha, uint nnz, uint nU, uint nI, uint nF)
{
  size_t i, j, k;
  size_t ij, item;
  double tmp;
  double *restrict m1;
  double *restrict m2;

  for (size_t it = n; it--; ) {
    memcpy(Rt, R, sizeof(double) * nI*nF);
    memcpy(Lt, L, sizeof(double) * nU*nF);

    for (ij = 0; ij < nnz; ++ij) {
      i = A->row[ij]; j = A->col[ij];
      m1 = &L[i*nF];
      m2 = &R[j*nF];
      tmp = 0;
      for (k = 0; k < nF; ++k) {
        tmp += Lt[i*nF+k] * Rt[j*nF + k];
      }
      tmp = A->val[ij] - tmp;
      for (k = 0; k < nF; ++k) {
        m1[k] += alpha * 2 * tmp * Rt[j*nF + k];
        m2[k] += alpha * 2 * tmp * Lt[i*nF + k];
      }
    }

  }
  matrix_mult_LR(nU, nI, nF);

  for (i = 0; i < nnz; ++i)
    B[A->row[i]*nI + A->col[i]] = 0;
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
main(int argc, char **argv)
{
  FILE *fp;

  argv0 = argv[0];
  if (argc != 2)
    usage();

  if (NULL == (fp = fopen(argv[1], "r")))
    die("unable to open file: \'%s\'\n", argv[1]);

  uint N   = parse_uint(fp);
  double a = parse_double(fp);
  uint nF  = parse_uint(fp);
  uint nU  = parse_uint(fp);
  uint nI  = parse_uint(fp);
  uint nnz = parse_uint(fp);

  A = new_csr_matrix("A", nnz);
  for (size_t ix = 0; ix < nnz; ++ix) {
    A->row[ix] = parse_uint(fp);
    A->col[ix] = parse_uint(fp);
    A->val[ix] = parse_double(fp);
  }

  if (0 != fclose(fp))
    die("unable to flush file stream.\n");

  L  = new_matrix("L", nU, nF);
  Lt = new_matrix("Lt", nU, nF);
  R  = new_matrix("R", nI, nF);
  Rt = new_matrix("Rt", nI, nF);
  B  = new_matrix("B", nU, nI);

  random_fill_LR(nU, nI, nF);

  run(N, a, nnz, nU, nI, nF);

  delete_matrix(B);
  delete_matrix(Rt);
  delete_matrix(R);
  delete_matrix(Lt);
  delete_matrix(L);
  delete_csr_matrix(A);

  return 0;
}
