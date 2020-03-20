#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

struct csr {
  unsigned int *row;
  unsigned int *col;
  double *val;
} __attribute((aligned(64)));

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
unsigned int
parse_uint(FILE *fp)
{
  unsigned int value;

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
new_matrix(unsigned int l, unsigned int c)
{
  double *m = (double *) malloc(sizeof(double) * l * c);
  if (!m)
    die("unable to allocate memory for matrix.\n");

  memset(m, '\x00', sizeof(double)*l*c);

  return m;
}

void
delete_matrix(double *m)
{
  if (m)
    free(m);
}

void
print_matrix(const double *m, unsigned int l, unsigned int c)
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
new_csr_matrix(unsigned int num_nz)
{
  struct csr *m;

  m = (struct csr *) malloc(sizeof(struct csr));
  if (!m)
    die("unable to allocate memory for sparse matrix.\n");

  m->row = (unsigned int *) malloc(sizeof(unsigned int) * num_nz);
  m->col = (unsigned int *) malloc(sizeof(unsigned int) * num_nz);
  m->val = (double *) malloc(sizeof(double) * num_nz);

  if (!m->row || !m->col || !m->val)
    die("unable to allocate memory for sparse matrix.\n");

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
random_fill_LR(unsigned int nU, unsigned int nI, unsigned int nF)
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
matrix_mult_LR(unsigned int nU, unsigned int nI, unsigned int nF)
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
matrix_fact_B(unsigned int n, double alpha, unsigned int nnz,
    unsigned int nU, unsigned int nI, unsigned int nF)
{
  size_t i, j, ij, k;
  double tmp;
  double *restrict m1;
  double *restrict m2;
  double *restrict mres; 

  do {
    memcpy(Rt, R, sizeof(double) * nI*nF);
    memcpy(Lt, L, sizeof(double) * nU*nF);

    for (ij = 0; ij < nnz; ++ij) {
      i = A->row[ij]; j = A->col[ij];
      m1 = &L[i*nF];
      m2 = &R[j*nF];
      tmp = A->val[ij] - B[i*nI + j];
      for (k = 0; k < nF; ++k) {
        m1[k] += alpha * 2 * tmp * Rt[j*nF + k];
        m2[k] += alpha * 2 * tmp * Lt[i*nF + k];
      }
    }

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
  } while(--n);
}

void
recommend(unsigned int nnz, unsigned int l, unsigned int c)
{
  size_t i, j;
  size_t item;
  double max;

  for (i = 0; i < nnz; i++)
    B[A->row[i]*c + A->col[i]] = 0;

  for (i = 0; i < l; ++i) {
    item = 0; max = -1.0;
    for (j = 0; j < c; ++j) {
      if (B[i*c + j] > max) {
        max = B[i*c + j];
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

  unsigned int N = parse_uint(fp);
  double alpha = parse_double(fp);
  unsigned int numF = parse_uint(fp);
  unsigned int numU = parse_uint(fp);
  unsigned int numI = parse_uint(fp);
  unsigned int nnz  = parse_uint(fp);

  A = new_csr_matrix(nnz);
  for (size_t ix = 0; ix < nnz; ++ix) {
    A->row[ix] = parse_uint(fp);
    A->col[ix] = parse_uint(fp);
    A->val[ix] = parse_double(fp);
  }

  if (0 != fclose(fp))
    die("unable to flush file stream.\n");

  L  = new_matrix(numU, numF);
  Lt = new_matrix(numU, numF);
  R  = new_matrix(numI, numF);
  Rt = new_matrix(numI, numF);
  B  = new_matrix(numU, numI);

  random_fill_LR(numU, numI, numF);
  matrix_mult_LR(numU, numI, numF);
  matrix_fact_B(N, alpha, nnz, numU, numI, numF);
  recommend(nnz, numU, numI);

  delete_matrix(B);
  delete_matrix(Rt);
  delete_matrix(R);
  delete_matrix(Lt);
  delete_matrix(L);

  delete_csr_matrix(A);
  return 0;
}
