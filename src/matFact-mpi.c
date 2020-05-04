#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <mpi.h>

#define DEBUG  0
#define RAND01 ((double)rand() / (double)RAND_MAX)

typedef unsigned int uint;

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
  for (size_t ix = 0; ix < size; ++ix) {
    ans += m1[ix] * m2[ix];
  }
  return ans;
}

void
solve()
{
  size_t i, j, k;
  double tmp, max;
  double *m1, *m2;
  double *aux_ptr;
  uint best[nU];
  int low, high;

  while (N--) {
    MPI_Barrier(MPI_COMM_WORLD);

    aux_ptr = L; L = Lt; Lt = aux_ptr;
    aux_ptr = R; R = Rt; Rt = aux_ptr;

    low = rank * nU / nproc;
    high = (rank + 1) * nU / nproc;

    for (i = 0, m1 = &A[i*nI], m2 = &B[i*nI];
        i < nU; ++i, m1 += nI, m2 += nI)
      for (j = 0; j < nI; ++j)
        if (m1[j])
          m2[j] = dot_prod(&Lt[i*nF], &Rt[j*nF], nF);

    // Update L
    for (i = low; i < high; ++i) {
      // For each L_{i,*} we need the entire matrix R
      for (k = 0; k < nF; ++k) {
        tmp = 0;
        for (j = 0; j < nI; ++j) {
          if (!A[i*nI+j]) continue;
          tmp += 2 * (A[i*nI+j] - B[i*nI+j]) * (-Rt[j*nF+k]);
        }
        L[i*nF + k] = Lt[i*nF + k] - a * tmp;
      }
    }

    low = rank * nI / nproc;
    high = (rank + 1) * nI / nproc;
    // Update R
    for (j = low; j < high; ++j) {
      // For each R_{*,j} we need the entire matrix L
      for (k = 0; k < nF; ++k) {
        tmp = 0;
        for (i = 0; i < nU; ++i) {
          if (!A[i*nI+j]) continue;
          tmp += 2 * (A[i*nI+j] - B[i*nI+j]) * (-Lt[i*nF+k]);
        }
        R[j*nF + k] = Rt[j*nF + k] - a * tmp;
      }
    }

    low = rank * nU / nproc;
    high = (rank + 1) * nU / nproc;

    // FIXME: This code is ugly
    int recvcnts[nproc];
    int displs[nproc];
    for (i = 0; i < nproc; ++i) {
      int l = i * nU / nproc;
      int h = (i + 1) * nU / nproc;
      recvcnts[i] = ((h-l)*nF);
      displs[i] = l*nF;
    }
    int size = (high-low)*nF;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgatherv(&L[low*nF], size, MPI_DOUBLE, L, recvcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    // FIXME: This code is ugly
    low = rank * nI / nproc;
    high = (rank + 1) * nI / nproc;
    size = (high-low)*nF;
    for (i = 0; i < nproc; ++i) {
      int l = i * nI / nproc;
      int h = (i + 1) * nI / nproc;
      recvcnts[i] = ((h-l)*nF);
      displs[i] = l*nF;
    }
 
    MPI_Allgatherv(&R[low*nF], size, MPI_DOUBLE, R, recvcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (0) {
      printf("[DEBUG] {rank=%d,L=\n", rank);
      matrix_print(L, nU, nF);
      printf("}\n");
      printf("[DEBUG] {rank=%d,Lt=\n", rank);
      matrix_print(Lt, nU, nF);
      printf("}\n");

      printf("[DEBUG] {rank=%d,R=\n", rank);
      matrix_print(R, nI, nF);
      printf("}\n");
      printf("[DEBUG] {rank=%d,Rt=\n", rank);
      matrix_print(Rt, nI, nF);
      printf("}\n");
    }
  }

  low = rank * nU / nproc;
  high = (rank + 1) * nU / nproc;
  if (0 == rank) {
    for (i = low; i < nU; ++i) {
      max = 0;
      m1 = &L[i * nF];
      for (j = 0; j < nI; ++j) {
        m2 = &R[j * nF];
        if (A[i*nI + j]) continue;
        tmp = 0;
        for (k = 0; k < nF; ++k) tmp += m1[k] * m2[k];
        if (tmp > max) {
          max = tmp;
          best[i] = j;
        }
      }
    }

    for (i = 0; i < nU; ++i) printf("%u\n", best[i]);
  }  
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (DEBUG) printf("[DEBUG] {rank=%d,nproc=%d}\n", rank, nproc);

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
    while (nnz--) {
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

  if (DEBUG) printf("[DEBUG] {rank=%d}->{N=%u,a=%lf,nF=%u,nU=%u,nI=%u}\n", rank, N, a, nF, nU, nI);

  if (NULL == A) {
    matrix_init(&A, nU, nI);
    matrix_init(&B, nU, nI);
  }

  MPI_Bcast(A, nU*nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  matrix_init(&L, nU, nF);
  matrix_init(&Lt, nU, nF);
  matrix_init(&R, nI, nF);
  matrix_init(&Rt, nI, nF);

  random_fill_LR();
  if (DEBUG) {
    printf("[DEBUG] {rank=%d,L=\n", rank);
    matrix_print(L, nU, nF);
    printf("}\n");

    printf("[DEBUG] {rank=%d,R=\n", rank);
    matrix_print(R, nI, nF);
    printf("}\n");

    printf("[DEBUG] {rank=%d,A=\n", rank);
    matrix_print(A, nU, nI);
    printf("}\n");
  }
  solve();

  matrix_destroy(Rt);
  matrix_destroy(R);
  matrix_destroy(Lt);
  matrix_destroy(L);
  matrix_destroy(A);
  matrix_destroy(B);

  MPI_Finalize();
  return 0;
}
