#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <mpi.h>

#define DEBUG  0
#define RAND01 ((double)rand() / (double)RAND_MAX)

typedef unsigned int uint;

struct csr {
  uint *row;
  uint *col;
  double *val;
} __attribute((aligned(64)));

/* globals */
static char *argv0 = NULL;  /* name of the binary */

static int rank  = 0;  /* calling process id  */
static int nproc = 0;  /* number of processes */

static uint N  = 0;  /* iteration count */
static uint nF = 0;  /* number of feats */
static uint nU = 0;  /* number of users */
static uint nI = 0;  /* number of items */

static double a = 0;  /* alpha constant */

static double *B  = NULL;  /* evaluation matrix */
static double *A  = NULL;  /* evaluation matrix */
static double *Lt = NULL;  /* users-feats matrix (prev it) */ 
static double *L  = NULL;  /* users-feats matrix (curr it) */
static double *Rt = NULL;  /* feats-items matrix (prev it) */
static double *R  = NULL;  /* feats-items matrix (curr it) */

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
  die("usage:\n\t%s <input-file>\n", argv0);
}

/* parsing helpers */
uint
parse_uint(FILE *fp)
{
  uint val = 0;

  if (1 != fscanf(fp, "%u", &val)) {
    die("[ERROR] %s: unable to parse value.\n", "parse_uint");
  }
  return val;
}

double
parse_double(FILE *fp)
{
  double val = 0;

  if (1 != fscanf(fp, "%lf", &val)) {
    die("[ERROR] %s: unable to parse value.\n", "parse_double");
  }
  return val;
}

/* matrix helpers */
void
matrix_init(double *matrix[], uint l, uint c)
{
  *matrix = (double *) malloc(sizeof(double) * l * c);
  if (NULL == *matrix) {
    die("[ERROR] %s: unable to allocate matrix\n", "matrix_init");
  }
  memset(*matrix, '\x00', sizeof(double) * l * c);
}

void
matrix_destroy(double matrix[]) {
  if (NULL != matrix) {
    free(matrix);
  }
}

void
matrix_print(const double matrix[], const uint l, const uint c)
{
  size_t i, j;

  for (i = 0; i < l; ++i) {
    for (j = 0; j < c; ++j) {
      printf("%1.6lf ", matrix[i * c + j]);
    }
    printf("\n");
  }
}

void
csr_matrix_init(struct csr **matrix, uint nnz)
{
  *matrix = (struct csr *) malloc(sizeof(struct csr));
  if (NULL == *matrix) {
    die("[ERROR] %s: unable to allocate matrix\n", "csr_matrix_init");
  }

  (*matrix)->row = (uint *) malloc(sizeof(uint) * nnz);
  (*matrix)->col = (uint *) malloc(sizeof(uint) * nnz);
  (*matrix)->val = (double *) malloc(sizeof(double) * nnz);
  if (NULL == (*matrix)->row || NULL == (*matrix)->col ||
      NULL == (*matrix)->val) {
    die("[ERROR] %s: unable to allocate vector\n", "csr_matrix_init");
  }
  memset((*matrix)->row, '\x00', sizeof(uint) * nnz);
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

/* required initilizer */
void
random_fill_LR()
{
  size_t i, j;

  srand(0);

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

double
dot_prod(double m1[], double m2[], size_t size)
{
  double ans = 0;
  for (size_t k = 0; k < size; ++k) {
    ans += m1[k] * m2[k];
  }
  return ans;
}

void
solve()
{
  uint *best;
  size_t i, j, k;
  double tmp, max;
  double *m1, *m2;
  int low_L, high_L;
  int low_R, high_R;
  int chunk_size_L, chunk_size_R;
  int recvcnts_L[nproc], displs_L[nproc];
  int recvcnts_R[nproc], displs_R[nproc];

  low_L = rank * nU / nproc; high_L = (rank + 1) * nU / nproc;
  low_R = rank * nI / nproc; high_R = (rank + 1) * nI / nproc;
  
  chunk_size_L = (high_L - low_L) * nF;
  chunk_size_R = (high_R - low_R) * nF;

  for (i = 0; i < nproc; ++i) {
    int l = i * nU / nproc;
    int h = (i + 1) * nU / nproc;
    recvcnts_L[i] = (h - l) * nF; displs_L[i] = l * nF;
    
    l = i * nI / nproc;
    h = (i + 1) * nI / nproc;
    recvcnts_R[i] = (h - l) * nF; displs_R[i] = l * nF;
  }

  while (N--) {
    memcpy(Rt, R, sizeof(double) * nI * nF);
    memcpy(Lt, L, sizeof(double) * nU * nF);

    // Update L
    for (i = low_L; i < high_L; ++i) {
      for (j = 0; j < nI; ++j) {
        if (!A[i*nI + j]) continue;
        tmp = dot_prod(&Lt[i*nF], &Rt[j*nF], nF);
        tmp = a * 2 * (A[i*nI + j] - tmp);
        for (k = 0; k < nF; ++k)
          L[i*nF + k] += tmp * Rt[j*nF + k];
      }
    }

    MPI_Allgatherv(
        &L[low_L*nF],
        chunk_size_L,
        MPI_DOUBLE,
        L,
        recvcnts_L,
        displs_L,
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    // Update R
    for (i = 0; i < nU; ++i) {
      for (j = low_R; j < high_R; ++j) {
        if (!A[i*nI + j]) continue;
        tmp = dot_prod(&Lt[i*nF], &Rt[j*nF], nF);
        tmp = a * 2 * (A[i*nI + j] - tmp);
        for (k = 0; k < nF; ++k)
          R[j*nF + k] += tmp * Lt[i*nF + k];
      }
    }

    MPI_Allgatherv(
        &R[low_R*nF],
        chunk_size_R,
        MPI_DOUBLE,
        R,
        recvcnts_R,
        displs_R,
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
  }

  uint ex;
  uint best_chunk[high_L - low_L];

  for (i = low_L, ex = 0; i < high_L; ++i, ++ex) {
    max = 0;
    m1 = &L[i * nF];
    for (j = 0; j < nI; ++j) {
      m2 = &R[j * nF];
      if (A[i*nI + j]) continue;
      tmp = 0;
      for (k = 0; k < nF; ++k)
        tmp += m1[k] * m2[k];
      if (tmp > max) {
        max = tmp;
        best_chunk[ex] = j;
      }
    }
  }

  for (i = 0; i < nproc; ++i) {
    int l = i * nU / nproc;
    int h = (i + 1) * nU / nproc;
    recvcnts_L[i] = h - l;
    displs_L[i] = l;
  }

  if (0 == rank) best = (uint *) malloc(sizeof(uint) * nU);

  MPI_Gatherv(
      best_chunk,
      high_L - low_L,
      MPI_INT,
      best,
      recvcnts_L,
      displs_L,
      MPI_INT,
      0,
      MPI_COMM_WORLD
  );

  if (0 == rank) {
    for (i = 0; i < nU; ++i) printf("%u\n", best[i]);
    free(best);
  }
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (0 == rank) {
    FILE *fp;
    size_t i, j, nnz;

    argv0 = argv[0];
    if (2 != argc) {
      MPI_Finalize();
      usage();
    }

    if (NULL == (fp = fopen(argv[1], "r"))) {
      die("[ERROR] %s: unable to open file \'%s\'\n", "main", argv[1]);
    }
    
    N   = parse_uint(fp);
    a   = parse_double(fp);
    nF  = parse_uint(fp);
    nU  = parse_uint(fp);
    nI  = parse_uint(fp);
    nnz = parse_uint(fp);

    matrix_init(&B, nU, nI);
    matrix_init(&A, nU, nI);

    /**
     * parse matrix A
     * TODO: distribute matrix A
     */
    for (size_t ij = 0; ij < nnz; ++ij) {
#if 0
      A->row[ij] = parse_uint(fp);
      A->col[ij] = parse_uint(fp);
      A->val[ij] = parse_uint(fp);
#endif
      i = parse_uint(fp);
      j = parse_uint(fp);
      A[i* nI + j] = parse_double(fp);
    }

    if (0 != fclose(fp)) {
      die("[ERROR] %s: unable to flush file stream\n", "main");
    }
  }

  /* send parameters to nodes */
  uint vec[4] = {N, nF, nU, nI};
  MPI_Bcast(vec, 4, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  N = vec[0]; nF = vec[1]; nU = vec[2]; nI = vec[3];

  if (NULL == A) {
    matrix_init(&A, nU, nI);
  }

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
