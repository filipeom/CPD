#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
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
uint32_t
parse_uint(FILE *fp)
{
  uint32_t value;

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
double** 
new_matrix(uint32_t l, uint32_t c)
{
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
delete_matrix(double **m, uint32_t l)
{
  size_t i;

  for (i = 0; i < l; i++)
    free(m[i]);
  free(m);
}

void 
print_matrix(double **m, uint32_t l, uint32_t c)
{
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
random_fill_LR(uint32_t nU, uint32_t nI, uint32_t nF)
{
  size_t i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L[i][j] = RAND01 / (double) nF;

  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R[j][i] = RAND01 / (double) nF;
}

void
matrix_mult(uint32_t nU, uint32_t nI, uint32_t nF)
{
  double sum;
  size_t i, j, k;

  for (i = 0; i < nU; i++) {
    for (j = 0; j < nI; j++) {
      sum = 0;
      for (k = 0; k < nF; k++) {
        sum += L[i][k] * R[j][k];
      }
      B[i][j] = sum;
    }
  }
}

void
matrix_fact_B(uint32_t n, double a, uint32_t nU, uint32_t nI,
    uint32_t nF)
{
  size_t i, j, k;
  double **tmp, sum;

  do {
    tmp = L; L = Lt; Lt = tmp;
    tmp = R; R = Rt; Rt = tmp;

    for (i = 0; i < nU; i++) {
      for (k = 0; k < nF; k++) {
        sum = 0;
        for (j = 0; j < nI; j++) {
          if (A[i][j]) sum += 2 * (A[i][j] - B[i][j]) * (-Rt[j][k]);
        }
        L[i][k] = Lt[i][k] - (a * sum);
      }
    }

    for (j = 0; j < nI; j++) {
      for (k = 0; k < nF; k++) {
        sum = 0;
        for (i = 0; i < nU; i++) {
          if (A[i][j]) sum += 2 * (A[i][j] - B[i][j]) * (-Lt[i][k]);
        }
        R[j][k] = Rt[j][k] - (a * sum);
      }
    }

    matrix_mult(nU, nI, nF);
  } while(--n);
}

void
recommend(uint32_t l, uint32_t c)
{
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
main(int argc, char **argv)
{
  FILE *fp;
  double alpha;
  size_t i, j;
  uint32_t N, lines;
  uint32_t numU, numI, numF;

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
  while(lines--) {
    i = parse_uint(fp);
    j = parse_uint(fp);
    A[i][j] = parse_double(fp);
  }

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

  delete_matrix(B, numU);
  delete_matrix(Rt, numI); 
  delete_matrix(R, numI);
  delete_matrix(Lt, numU);
  delete_matrix(L, numU);
  delete_matrix(A, numU);
  return 0;
}
