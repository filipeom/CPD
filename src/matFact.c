#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

/* globals */
static double *A  = NULL;
static double *B  = NULL;
static double *Lt = NULL;
static double *L  = NULL;
static double *Rt = NULL;
static double *R  = NULL;

static char *argv0;

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
  if (NULL == m)
    die("unable to allocate memory for matrix.\n");
  
  memset(m, '\x00', sizeof(double)*l*c);

  return m;
}

void 
delete_matrix(double *m)
{
  if (NULL != m) {
    free(m);
    m = NULL;
  }
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
matrix_mult(unsigned int nU, unsigned int nI, unsigned int nF)
{
  double sum;
  size_t i, j, k;

  for (i = 0; i < nU; ++i) {
    for (j = 0; j < nI; ++j) {
      sum = 0;
      for (k = 0; k < nF; ++k) {
        sum += L[i*nF + k] * R[j*nF + k];
      }
      B[i*nI + j] = sum;
    }
  }
}

void
matrix_fact_B(unsigned int n, double a, unsigned int nU, unsigned int nI,
    unsigned int nF)
{
  size_t i, j, k;
  double *tmp, sum;

  do {
    tmp = L; L = Lt; Lt = tmp;
    tmp = R; R = Rt; Rt = tmp;

    for (i = 0; i < nU; ++i) {
      for (k = 0; k < nF; ++k) {
        sum = 0;
        for (j = 0; j < nI; ++j) {
          if (A[i*nI + j]) sum += 2 * (A[i*nI + j] - B[i*nI + j]) * (-Rt[j*nF + k]);
        }
        L[i*nF + k] = Lt[i*nF + k] - (a * sum);
      }
    }

    for (j = 0; j < nI; ++j) {
      for (k = 0; k < nF; ++k) {
        sum = 0;
        for (i = 0; i < nU; ++i) {
          if (A[i*nI + j]) sum += 2 * (A[i*nI + j] - B[i*nI + j]) * (-Lt[i*nF + k]);
        }
        R[j*nF + k] = Rt[j*nF + k] - (a * sum);
      }
    }

    for (i = 0; i < nU; ++i) {
      for (j = 0; j < nI; ++j) {
        sum = 0;
        for (k = 0; k < nF; ++k) {
          sum += L[i*nF + k] * R[j*nF + k];
        }
        B[i*nI + j] = sum;
      }
    }
  } while(--n);
}

void
recommend(unsigned int l, unsigned int c)
{
  double max;
  size_t i, j, item;

  for (i = 0; i < l; ++i) {
    item = 0; max = -1.0;
    for (j = 0; j < c; ++j) {
      if ((B[i*c + j] > max) && !A[i*c + j]) {
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
  double alpha;
  size_t i, j;
  unsigned int N, lines;
  unsigned int numU, numI, numF;

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
    A[i*numI + j] = parse_double(fp);
  } while(--lines);

  if (0 != fclose(fp))
    die("unable to flush file stream.\n");

  L  = new_matrix(numU, numF);
  Lt = new_matrix(numU, numF);
  R  = new_matrix(numI, numF);
  Rt = new_matrix(numI, numF);
  B  = new_matrix(numU, numI);

  random_fill_LR(numU, numI, numF);
  matrix_mult(numU, numI, numF);
  matrix_fact_B(N, alpha, numU, numI, numF);
  recommend(numU, numI);

  delete_matrix(B);
  delete_matrix(Rt); 
  delete_matrix(R);
  delete_matrix(Lt);
  delete_matrix(L);
  delete_matrix(A);
  return 0;
}
