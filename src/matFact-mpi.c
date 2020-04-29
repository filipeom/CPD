#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <mpi.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

typedef unsigned int uint;

struct csr {
  uint *row;
  uint *col;
  double *val;
} __attribute__((aligned(64)));

/* globals */
static char *argv0 = NULL;

static int id = 0;
static int nproc = 0;

static uint N   = 0;
static double a = 0; 
static uint nF  = 0;
static uint nU  = 0;
static uint nI  = 0;

static double *A  = NULL;
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

  if (1 != fscanf(fp, "%u", &value)) {
    die("unable to parse integer.\n");
  }
  return value;
}

double
parse_double(FILE *fp)
{
  double value;

  if (1 != fscanf(fp, "%lf", &value)) {
    die("unable to parse double.\n");
  }
  return value;
}

/* matrix helpers */
void
matrix_init(double *matrix[], uint l, uint c)
{
  *matrix = (double *) malloc(sizeof(double) * l * c);
  if (NULL == *matrix) {
    die("unable to allocate memory for matrix.\n");
  }

  memset(*matrix, '\x00', sizeof(double) * l * c);
}

void
matrix_destroy(double *matrix)
{
  if (NULL != matrix) {
    free(matrix);
  }
}

void
print_matrix(const double *matrix, uint l, uint c)
{
  size_t i, j;

  for (i = 0; i < l; ++i) {
    for (j= 0; j < c; ++j) {
      printf("%1.6lf ", matrix[i * c + j]);
    }
    printf("\n");
  }
}

void
csr_matrix_init(struct csr **matrix, uint nnz, uint nU)
{
  *matrix = (struct csr *) malloc(sizeof(struct csr));
  if (NULL == *matrix) {
    die("unable to allocate memory for sparse matrix.\n");
  }

  (*matrix)->row = (uint *) malloc(sizeof(uint) * (nU + 1));
  (*matrix)->col = (uint *) malloc(sizeof(uint) * nnz);
  (*matrix)->val = (double *) malloc(sizeof(double) * nnz);
  if (NULL == (*matrix)->row || NULL == (*matrix)->col ||
      NULL == (*matrix)->val) {
    die("unable to allocate memory for sparse matrix.\n");
  }

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
random_fill_LR()
{
  size_t i, j;

  srandom(0);

  for (i = 0; i < nU; ++i) {
    for (j = 0; j < nF; ++j) {
      L[i * nF + j] = RAND01 / (double) nF;
    }    
  }
  for (i = 0; i < nF; ++i) {
    for (j = 0; j < nI; ++j) {
      R[j * nF + i] = RAND01 / (double) nF;
    }    
  }
}

void
solve()
{
  size_t i, j, k;
  double tmp, max;
  double *restrict m1, *restrict mt1;
  double *restrict m2, *restrict mt2;
  uint best[nU];

  uint low = id * nU / nproc;
  uint high = (id + 1) * nU / nproc;

  for (size_t it = N; it--; ) {
    memcpy(Rt, R, sizeof(double) * nI * nF);
    memcpy(Lt, L, sizeof(double) * nU * nF);
    for (i = low, m1 = &L[i * nF], mt1 = &Lt[i * nF]; i < high; 
        ++i, m1 += nF, mt1 += nF) {
      for (j = 0; j < nI; ++j) {
        if (A[i*nI + j]) {
          m2 = &R[j * nF];
          mt2 = &Rt[j * nF];
          tmp = 0;
          for (k = 0; k < nF; ++k) {
            tmp += mt1[k] * mt2[k];
          }
          tmp = a * 2 * (A[i*nI + j] - tmp);
          for (k = 0; k < nF; ++k) {
            m1[k] += tmp * mt2[k];
            m2[k] += tmp * mt1[k];
          }
        }
      }
    }
  }

  for (i = low, m1 = &L[i * nF]; i < high; ++i, m1 += nF) {
    max = 0;
    for (j = 0, m2 = &R[j * nF]; j < nI; ++j, m2 += nF) {
      if (!A[i*nI + j]) {
        tmp = 0;
        for (k = 0; k < nF; ++k) {
          tmp += m1[k] * m2[k];
        }
        if (tmp > max) {
          max = tmp;
          best[i] = j;
        }
      } 
    }
  }

  for (i = low; i < high; ++i) {
      printf("%u\n", best[i]);
  }
}

/* main */
int
main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (0 == id) {
    FILE *fp;
    size_t i, j, nnz;

    argv0 = argv[0] ;
    if (2 != argc) {
      usage();
      MPI_Finalize();
    }
    
    if (NULL == (fp = fopen(argv[1], "r"))) {
      die("unable to open file: \'%s\'\n", argv[1]);
    }

    N   = parse_uint(fp);
    a   = parse_double(fp);
    nF  = parse_uint(fp);
    nU  = parse_uint(fp);
    nI  = parse_uint(fp);
    nnz = parse_uint(fp);

    matrix_init(&A, nU, nI);

    for (size_t ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      A[i*nI + j] = parse_double(fp);
    }

    if (0 != fclose(fp)) {
      die("unable to flush file stream.\n");
    }
  } 

  // TODO: Send one vector of unsigned => 2.5x less comms
  MPI_Bcast(&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nF, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nU, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nI, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (NULL == A)
    matrix_init(&A, nU, nI);
  
  MPI_Bcast(A, nU*nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  matrix_init(&L, nU, nF);
  matrix_init(&Lt, nU, nF);
  matrix_init(&R, nI, nF);
  matrix_init(&Rt, nI, nF);

  random_fill_LR();
  solve();

  matrix_destroy(Rt);
  matrix_destroy(R);
  matrix_destroy(Lt);
  matrix_destroy(L);
  matrix_destroy(A);

  MPI_Finalize();
  return 0;
}
