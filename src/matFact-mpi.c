#include <stdio.h>
#include <errno.h>
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
static int pid  = 0;  /* calling process id  */
static int nproc = 0;  /* number of processes */

static uint N   = 0;  /* iteration count */
static uint nF  = 0;  /* number of feats */
static uint nU  = 0;  /* number of users */
static uint nI  = 0;  /* number of items */
static uint nnz = 0;  /* number of !zero */

static double a = 0;  /* alpha constant */

static double *Lt = NULL;  /* users-feats matrix (prev it) */ 
static double *L  = NULL;  /* users-feats matrix (curr it) */
static double *Rt = NULL;  /* feats-items matrix (prev it) */
static double *R  = NULL;  /* feats-items matrix (curr it) */

static struct csr *Al = NULL;
static struct csr *Ac = NULL;

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

void *
xmalloc(size_t len)
{
  void *ptr;

  if (NULL == (ptr = malloc(len))) {
    die("malloc: %s\n", strerror(errno));
  }

  return ptr;
}

/* parsing helpers */
uint
parse_uint(FILE *fp)
{
  uint val;

  if (1 != fscanf(fp, "%u", &val)) {
    die("parse_uint: unable to parse value.\n");
  }

  return val;
}

double
parse_double(FILE *fp)
{
  double val;

  if (1 != fscanf(fp, "%lf", &val)) {
    die("parse_double: unable to parse value.\n");
  }

  return val;
}

/* matrix helpers */
double *
matrix_init(uint l, uint c)
{
  double *matrix;

  matrix = (double *) xmalloc(sizeof(double) * (l * c));

  memset(matrix, '\x00', sizeof(double) * (l * c));

  return matrix;
}

void
matrix_destroy(double *matrix[]) {
  if (NULL != *matrix)
    free(*matrix);

  *matrix = NULL;
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

struct csr *
csr_matrix_init(uint nrows, uint nnz)
{
  struct csr *matrix;

  matrix = (struct csr *) xmalloc(sizeof(struct csr));

  matrix->row = (uint *) xmalloc(sizeof(uint) * (nrows + 1));
  matrix->col = (uint *) xmalloc(sizeof(uint) * nnz);
  matrix->val = (double *) xmalloc(sizeof(double) * nnz);

  memset(matrix->row, '\x00', sizeof(uint) * (nrows + 1));
  memset(matrix->col, '\x00', sizeof(uint) * nnz);
  memset(matrix->val, '\x00', sizeof(double) * nnz);

  return matrix;
}

void
csr_matrix_destroy(struct csr **matrix)
{
  struct csr *m = *matrix;

  if (NULL != m->row && NULL != m->col &&
      NULL != m->val && NULL != m) {
    free(m->row);
    free(m->col);
    free(m->val);
    free(m);
  }

  *matrix = NULL;
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
  uint *best, *chunk;
  size_t i, j, k, jx, ix;
  double tmp, max;
  double *m, *mt;
  int low, low_L, low_R;
  int high, high_L, high_R;
  int chunk_L, chunk_R;
  int cnts_L[nproc], cnts_R[nproc];
  int offs_L[nproc], offs_R[nproc];
  
  if (0 == pid) {
    best = (uint *) xmalloc(sizeof(uint) * nU);
  }

  low_L   = pid * nU / nproc;
  low_R   = pid * nI / nproc; 
  high_L  = (pid + 1) * nU / nproc;
  high_R  = (pid + 1) * nI / nproc;
  chunk_L = (high_L - low_L) * nF;
  chunk_R = (high_R - low_R) * nF;

  chunk = (uint *) xmalloc(sizeof(uint) * (high_L - low_L));

  for (i = 0; i < nproc; ++i) {
    low       = i * nU / nproc;
    high      = (i + 1) * nU / nproc;
    cnts_L[i] = (high - low) * nF;
    offs_L[i] = low * nF;
    
    low       = i * nI / nproc;
    high      = (i + 1) * nI / nproc;
    cnts_R[i] = (high - low) * nF;
    offs_R[i] = low * nF;
  }

  while (N--) {
    memcpy(Rt, R, sizeof(double) * nI * nF);
    memcpy(Lt, L, sizeof(double) * nU * nF);

    // Update L
    for (i = low_L; i < high_L; ++i) {
      m = &L[i * nF];
      for (jx = Al->row[i]; jx < Al->row[i + 1]; jx++) {
        j = Al->col[jx];
        mt = &Rt[j * nF];
        tmp = dot_prod(&Lt[i*nF], mt, nF);
        tmp = a * 2 * (Al->val[jx] - tmp);
        for (k = 0; k < nF; ++k)
          m[k] += tmp * mt[k];
      }
    }

    MPI_Allgatherv(
        &L[low_L*nF],     /* src buffer */
        chunk_L,          /* length of data to send */
        MPI_DOUBLE,
        L,                /* recv buffer */
        cnts_L,           /* buffer with the recvcnt from each proc */
        offs_L,           /* offset where each proc should write to L */
        MPI_DOUBLE,
        MPI_COMM_WORLD    /* group of processes involved in this comm */
    );

    //MPI_Barrier(MPI_COMM_WORLD);

    // Update R
    for (j = low_R; j < high_R; ++j) {
      m = &R[j * nF];
      for (ix = Ac->row[j]; ix < Ac->row[j + 1]; ix++) {
        i = Ac->col[ix];
        mt = &Lt[i * nF];
        tmp = dot_prod(mt, &Rt[j*nF], nF);
        tmp = a * 2 * (Ac->val[ix] - tmp);
        for (k = 0; k < nF; ++k)
          m[k] += tmp * mt[k];
      }
    }

    MPI_Allgatherv(
        &R[low_R*nF],     /* src buffer */
        chunk_R,          /* length of data to send */
        MPI_DOUBLE,
        R,                /* recv buffer */
        cnts_R,           /* buffer with the recvcnt from each proc */
        offs_R,           /* offset where each proc should write to R */
        MPI_DOUBLE,
        MPI_COMM_WORLD    /* group of processes involved in this comm */
    );

    MPI_Barrier(MPI_COMM_WORLD);
  } /* end while */

  for (i = low_L, ix = 0; i < high_L; ++i, ++ix) {
    max = 0;
    jx = Al->row[i];
    for (j = 0; j < nI; ++j) {
      if ((Al->row[i + 1] - Al->row[i]) &&
          (Al->row[i + 1] > jx) &&
          (Al->col[jx] == j)) {
        jx++;
      } else {
        tmp = dot_prod(&L[i * nF], &R[j * nF], nF);
        if (tmp > max) {
          max = tmp;
          chunk[ix] = j;
        }
      }
    }
  }

  for (i = 0; i < nproc; ++i) {
    low = i * nU / nproc;
    high = (i + 1) * nU / nproc;
    cnts_L[i] = high - low;
    offs_L[i] = low;
  }


  MPI_Gatherv(
      chunk,
      high_L - low_L,
      MPI_INT,
      best,
      cnts_L,
      offs_L,
      MPI_INT,
      0,
      MPI_COMM_WORLD
  );

  if (0 == pid) {
    for (i = 0; i < nU; ++i) printf("%u\n", best[i]);
    free(best);
    best = NULL;
  }

  free(chunk);
  chunk = NULL;
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (2 != argc) {
    MPI_Finalize();
    die("usage:\n\t%s <input-file>\n", argv[0]);
  }

  if (0 == pid) {
    FILE *fp;
    size_t i, j;
    uint sum, tmp, lst;

    if (NULL == (fp = fopen(argv[1], "r"))) {
      die("main: unable to open file \'%s\'\n", argv[1]);
    }
    
    N   = parse_uint(fp);
    a   = parse_double(fp);
    nF  = parse_uint(fp);
    nU  = parse_uint(fp);
    nI  = parse_uint(fp);
    nnz = parse_uint(fp);

    Al = csr_matrix_init(nU, nnz);
    Ac = csr_matrix_init(nI, nnz);

    /*
     * parse matrix A
     * TODO: distribute matrix A
     */
    for (uint ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      Al->row[i+1] += 1;
      Ac->row[j] += 1;
      Al->col[ij] = j;
      Al->val[ij] = parse_double(fp);
    }

    if (0 != fclose(fp)) {
      die("main: unable to flush file stream\n");
    }

    for (i = 1; i <= nU; ++i) {
      Al->row[i] += Al->row[i - 1];
    }

    /* CSR to CSC */
    for (j = 0, sum = 0; j < nI; ++j) {
      tmp = Ac->row[j];
      Ac->row[j] = sum;
      sum += tmp;
    }
    Ac->row[nI] = nnz;
    
    for (i = 0; i < nU; ++i) {
      for (size_t jx = Al->row[i]; jx < Al->row[i + 1]; ++jx) {
        uint c = Al->col[jx];
        uint d = Ac->row[c];

        Ac->col[d] = i;
        Ac->val[d] = Al->val[jx];
        Ac->row[c]++;
      }
    }

    for (j = 0, lst = 0; j <= nI; ++j) {
      tmp = Ac->row[j];
      Ac->row[j] = lst;
      lst = tmp;
    }
  }

  /* send parameters to nodes */
  uint vec[5] = {N, nF, nU, nI, nnz};
  MPI_Bcast(vec, 5, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  N   = vec[0];
  nF  = vec[1]; 
  nU  = vec[2];
  nI  = vec[3]; 
  nnz = vec[4];

  if ((NULL == Al) && (NULL == Ac)) {
    Al = csr_matrix_init(nU, nnz);
    Ac = csr_matrix_init(nI, nnz);
  }

  MPI_Bcast(Al->row, nU + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(Al->col, nnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(Al->val, nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(Ac->row, nI + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(Ac->col, nnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(Ac->val, nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  L  = matrix_init(nU, nF);
  Lt = matrix_init(nU, nF);
  R  = matrix_init(nI, nF);
  Rt = matrix_init(nI, nF);

  random_fill_LR();

  double secs;
  secs = - MPI_Wtime();
  solve();
  secs += MPI_Wtime();
  
  // Redirect stdout to file and get time on stderr
  if (0 == pid) fprintf(stderr, "Time = %12.6f sec\n", secs);

  matrix_destroy(&Rt);
  matrix_destroy(&R);
  matrix_destroy(&Lt);
  matrix_destroy(&L);

  csr_matrix_destroy(&Al);
  csr_matrix_destroy(&Ac);

  MPI_Finalize();
  return 0;
}
