#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <mpi.h>
#include <math.h>

#define TAG    0x42
#define DEBUG  0
#define RAND01 ((double)rand() / (double)RAND_MAX)

typedef unsigned int uint;

struct csr {
  uint   *row;
  uint   *col;
  double *val;
};

/* globals */
static int p     = 0;
static int nid   = 0;      /* calling node id  */
static int nproc = 0;      /* number of node */

static uint N   = 0;       /* iteration count */
static uint nF  = 0;       /* number of feats */
static uint nU  = 0;       /* number of users */
static uint nI  = 0;       /* number of items */
static uint nnz = 0;       /* number of !zero */

static double a = 0;       /* alpha constant */

static uint low_L  = 0;
static uint low_R  = 0;
static uint high_L = 0;
static uint high_R = 0;

static double *Lt = NULL;  /* users-feats matrix (prev it) */ 
static double *L  = NULL;  /* users-feats matrix (curr it) */
static double *Rt = NULL;  /* feats-items matrix (prev it) */
static double *R  = NULL;  /* feats-items matrix (curr it) */

static struct csr *A  = NULL;
static struct csr *At = NULL;

/* general helpers */
void
die(const char *err_str, ...)
{
  va_list ap;
  
  va_start(ap, err_str);
  vfprintf(stderr, err_str, ap);
  va_end(ap);
  /* abort all other */
  MPI_Abort(MPI_COMM_WORLD, 1);
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
  int low, high;
  int chunk_L, chunk_R;
  int cnts_L[nproc], cnts_R[nproc];
  int offs_L[nproc], offs_R[nproc];
  
  if (0 == nid) {
    best = (uint *) xmalloc(sizeof(uint) * nU);
  }

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
      for (jx = A->row[i]; jx < A->row[i + 1]; jx++) {
        j = A->col[jx];
        mt = &Rt[j * nF];
        tmp = dot_prod(&Lt[i*nF], mt, nF);
        tmp = a * 2 * (A->val[jx] - tmp);
        for (k = 0; k < nF; ++k)
          m[k] += tmp * mt[k];
      }
    }

    MPI_Allgatherv(
        &L[low_L*nF],     /* src buffer */
        chunk_L,          /* length of data to send */
        MPI_DOUBLE,       /* type of data to send */
        L,                /* recv buffer */
        cnts_L,           /* buffer with the recvcnt from each node */
        offs_L,           /* offset where each node should write to L */
        MPI_DOUBLE,       /* type of data to recv */
        MPI_COMM_WORLD    /* group of nodes involved in this comm */
    );

    // Update R
    for (j = low_R; j < high_R; ++j) {
      m = &R[j * nF];
      for (ix = At->row[j]; ix < At->row[j + 1]; ix++) {
        i = At->col[ix];
        mt = &Lt[i * nF];
        tmp = dot_prod(mt, &Rt[j*nF], nF);
        tmp = a * 2 * (At->val[ix] - tmp);
        for (k = 0; k < nF; ++k)
          m[k] += tmp * mt[k];
      }
    }

    MPI_Allgatherv(
        &R[low_R*nF],     /* src buffer */
        chunk_R,          /* length of data to send */
        MPI_DOUBLE,       /* type of data to send */
        R,                /* recv buffer */
        cnts_R,           /* buffer with the recvcnt from each node */
        offs_R,           /* offset where each node should write to R */
        MPI_DOUBLE,       /* type of data to recv */
        MPI_COMM_WORLD    /* group of nodes involved in this comm */
    );

    /* Synchronization necessary before each new iteration */
    MPI_Barrier(MPI_COMM_WORLD);
  } /* end while */

  for (i = low_L, ix = 0; i < high_L; ++i, ++ix) {
    max = 0;
    jx = A->row[i];
    for (j = 0; j < nI; ++j) {
      if ((A->row[i + 1] - A->row[i]) &&
          (A->row[i + 1] > jx) &&
          (A->col[jx] == j)) {
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
      chunk,              /* src buffer */
      high_L - low_L,     /* length of data to send */
      MPI_INT,            /* type of data to send */
      best,               /* recv buffer */
      cnts_L,             /* buffer with the recvcnt from each node */
      offs_L,             /* offset where root will start writing data from each node to recv buffer */
      MPI_INT,            /* type of data to recv */
      0,                  /* root nid */
      MPI_COMM_WORLD      /* group of nodes involved in this comm */
  );

  if (0 == nid) {
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
  int start, end;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &nid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* must be a perfect square for this */
  p = sqrt(nproc);

  if (0 == nid) {
    FILE *fp;
    size_t i, j;
    uint sum, tmp, lst;
    double v;

    if (2 != argc) {
      die("usage:\n\t%s <input-file>\n", argv[0]);
    }

    if (NULL == (fp = fopen(argv[1], "r"))) {
      die("main: unable to open file \'%s\'\n", argv[1]);
    }
    
    /* parse parameters */
    N   = parse_uint(fp);
    a   = parse_double(fp);
    nF  = parse_uint(fp);
    nU  = parse_uint(fp);
    nI  = parse_uint(fp);
    nnz = parse_uint(fp);

    /* root's slice */
    low_L   = nid * nU / nproc;
    low_R   = nid * nI / nproc; 
    high_L  = (nid + 1) * nU / nproc;
    high_R  = (nid + 1) * nI / nproc;

    /* send parameters to all nodes */
    for (int id = 1; id < nproc; ++id) {
      MPI_Send(&N,   1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&a,   1, MPI_DOUBLE,   id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nF,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nU,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nI,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nnz, 1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
    }

    A  = csr_matrix_init(nU, nnz);
    At = csr_matrix_init(nI, nnz);

    /* get current file ptr pos */
    long curr = ftell(fp);

    for (size_t ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      v = parse_double(fp); /* dummy parse */

      A->row[i + 1] += 1;
      At->row[j] += 1;
    }
    
    for (i = 1; i <= nU; ++i) {
      A->row[i] += A->row[i - 1];
    }

    /* send nnz offsets to all nodes */
    for (int id = 1; id < nproc; ++id) {
      MPI_Send(A->row, nU+1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
    }

    /* rewind to start of file */
    rewind(fp);
    /* set to start of nnz entries */
    fseek(fp, curr, SEEK_SET);

    int did = 1;
    for (size_t ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp); /* dummy parse */
      j = parse_uint(fp);
      v = parse_double(fp);

      A->col[ij] = j;
      A->val[ij] = v;

      if (ij == (A->row[((did + 1) * nU / nproc)] - 1) && nproc > 1) {
#if 0
        printf("I can now send data to thread %d\n", did);
        printf("{id=%d} -> {\n", did);
        int aux = did * nU / nproc;
        printf("\trow = %u, start = %u\n", aux, A->row[aux]);
        aux = (did + 1) * nU / nproc;
        printf("\trow = %u, end = %u\n", aux, A->row[aux]);
        printf("\ncurrent ij = %lu\n", ij);
        printf("}\n");
#endif
#if 1
        int low = did * nU / nproc;
        int high = (did + 1) * nU / nproc;
        start = A->row[low];
        end = A->row[high];
        MPI_Send(&A->col[start], end-start, MPI_UNSIGNED, did, TAG, MPI_COMM_WORLD);
        MPI_Send(&A->val[start], end-start, MPI_DOUBLE,   did, TAG, MPI_COMM_WORLD);
#endif
        did++;
      }
    }

    if (0 != fclose(fp)) {
      die("main: unable to flush file stream\n");
    }

    /* CSR to CSC */
    for (j = 0, sum = 0; j < nI; ++j) {
      tmp = At->row[j];
      At->row[j] = sum;
      sum += tmp;
    }
    At->row[nI] = nnz;
    
    uint col, dst;
    for (i = 0; i < nU; ++i) {
      for (size_t jx = A->row[i]; jx < A->row[i + 1]; ++jx) {
        col = A->col[jx];
        dst = At->row[col];

        At->col[dst] = i;
        At->val[dst] = A->val[jx];
        At->row[col]++;
      }
    }

    for (j = 0, lst = 0; j <= nI; ++j) {
      tmp = At->row[j];
      At->row[j] = lst;
      lst = tmp;
    }
  } else {
    MPI_Status status;

    MPI_Recv(&N,   1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&a,   1, MPI_DOUBLE,   0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nF,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nU,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nI,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nnz, 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);

    /* node's slice */
    low_L   = nid * nU / nproc;
    low_R   = nid * nI / nproc; 
    high_L  = (nid + 1) * nU / nproc;
    high_R  = (nid + 1) * nI / nproc;

    A  = csr_matrix_init(nU, nnz);
    At = csr_matrix_init(nI, nnz);

    MPI_Recv(A->row, nU + 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);

    start = A->row[low_L];
    end = A->row[high_L];

    MPI_Recv(&A->col[start], end-start, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&A->val[start], end-start, MPI_DOUBLE, 0,   TAG, MPI_COMM_WORLD, &status);
  }

#if 0
  printf("{rank=%d}->{N=%u,a=%lf,nF=%u,nU=%u,nI=%u,nnz=%u}\n",
      nid, N, a, nF, nU, nI, nnz);
#endif

  MPI_Bcast(At->row, nI + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(At->col, nnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(At->val, nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
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
  if (0 == nid) fprintf(stderr, "Time = %12.6f sec\n", secs);

  matrix_destroy(&Rt);
  matrix_destroy(&R);
  matrix_destroy(&Lt);
  matrix_destroy(&L);

  csr_matrix_destroy(&A);
  csr_matrix_destroy(&At);

  MPI_Finalize();
  return 0;
}
