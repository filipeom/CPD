#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define TAG    0x42
#define RAND01 ((double)rand() / (double)RAND_MAX)

typedef unsigned int uint;

struct csr {
  uint   *row;
  uint   *col;
  double *val;
};

/* globals - but only here */
static int p         = 0;
static int nid       = 0;  /* calling node id */
static int nproc     = 0;  /* number of nodes */
static int row_nid   = 0;
static int row_nproc = 0;
static int col_nid   = 0;
static int col_nproc = 0;

static MPI_Comm row_comm;
static MPI_Comm col_comm;

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
csr_matrix_init(uint nnz)
{
  struct csr *matrix;

  matrix = (struct csr *)  xmalloc(sizeof(struct csr));

  matrix->row = (uint *)   xmalloc(sizeof(uint)   * nnz);
  matrix->col = (uint *)   xmalloc(sizeof(uint)   * nnz);
  matrix->val = (double *) xmalloc(sizeof(double) * nnz);

  memset(matrix->row, '\x00', sizeof(uint)   * nnz);
  memset(matrix->col, '\x00', sizeof(uint)   * nnz);
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
random_fill_L()
{
  size_t i, j;

  srand(0);

  for (i = 0; i < low_L; ++i) {
    for (j = 0; j < nF; ++j) {
      rand();
    }
  }
  for (i = 0; i < (high_L - low_L); ++i) {
    for (j = 0; j < nF; ++j) {
      L[i * nF + j] = RAND01 / (double) nF;
    }
  }
  for (i = high_L; i < nU; ++i) {
    for (j = 0; j < nF; ++j) {
      rand();
    }
  }

}

void
random_fill_R()
{
  size_t i, j;

  for (i = 0; i < nF; ++i) {
    for (j = 0; j < low_R; ++j) {
      rand();
    }
    for (j = 0; j < (high_R - low_R); ++j) {
      R[j * nF + i] = RAND01 / (double) nF;
    }
    for (j = high_R; j < nI; ++j) {
      rand();
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
  size_t i, j, k, ij;
  double tmp;
  double *m1, *mt1;
  double *m2, *mt2;
  
  if (0 == nid) {
    best = (uint *) xmalloc(sizeof(uint) * nU);
  }

  chunk = (uint *) xmalloc(sizeof(uint) * (high_L - low_L));

  memcpy(Lt, L, sizeof(double) * (high_L - low_L) * nF);
  memcpy(Rt, R, sizeof(double) * (high_R - low_R) * nF);

  while (N--) {
    if (row_nid) memset(L, '\x00', sizeof(double) * (high_L - low_L) * nF);
    if (col_nid) memset(R, '\x00', sizeof(double) * (high_R - low_R) * nF);

    for (ij = 0; ij < nnz; ++ij) {
      i = A->row[ij]-low_L, j = A->col[ij]-low_R;
      m1 = &L[i * nF], mt1 = &Lt[i * nF];
      m2 = &R[j * nF], mt2 = &Rt[j * nF];
      tmp = a * 2 * (A->val[ij] - dot_prod(mt1, mt2, nF));
      for (k = 0; k < nF; ++k) {
        m1[k] += tmp * mt2[k];
        m2[k] += tmp * mt1[k];
      }
    }
    
    memset(Lt, '\x00', sizeof(double) * (high_L - low_L) * nF);

    MPI_Allreduce(
        L,
        Lt,
        (high_L - low_L) * nF,
        MPI_DOUBLE,
        MPI_SUM,
        row_comm
    );

    MPI_Barrier(row_comm);

    memset(Rt, '\x00', sizeof(double) * (high_R - low_R) * nF);

    MPI_Allreduce(
        R,
        Rt,
        (high_R - low_R) * nF,
        MPI_DOUBLE,
        MPI_SUM,
        col_comm
    );

    MPI_Barrier(col_comm);

    memcpy(L, Lt, sizeof(double) * (high_L - low_L) * nF);
    memcpy(R, Rt, sizeof(double) * (high_R - low_R) * nF);

    /* Synchronization necessary before each new iteration */
    MPI_Barrier(MPI_COMM_WORLD);
  } /* end while */

#if 0
  if (0 == row_nid) {
    printf("L in %d\n", nid);
    matrix_print(Lt, high_L - low_L, nF);
  }
  if (0 == col_nid) {
    printf("R in %d\n", nid);
    matrix_print(Rt, high_R - low_R, nF);
  }
#endif

// TODO: find max
#if 0
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
#endif

  if (0 == nid) {
    for (i = 0; i < nU; ++i) printf("%u\n", best[i]);
    free(best);
    best = NULL;
  }

  free(chunk);
  chunk = NULL;
}

int
get_idx(uint x, uint vsize, int psize)
{
  for (int id = 1; id <= psize; ++id)
    if ((x >= ((id - 1) * vsize / psize)) &&
        (x < (id * vsize / psize)))
      return id-1;
  return 0;
}

int
main(int argc, char* argv[])
{
  int color;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &nid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // FIXME: Support non-square grids
  p = sqrt(nproc);

  /* Create grid comms */
  color = floor(nid / p);
  MPI_Comm_split(MPI_COMM_WORLD, color, nid, &row_comm);
  MPI_Comm_rank(row_comm, &row_nid);
  MPI_Comm_size(row_comm, &row_nproc);

  color = (nid % p) + TAG;
  MPI_Comm_split(MPI_COMM_WORLD, color, nid, &col_comm);
  MPI_Comm_rank(col_comm, &col_nid);
  MPI_Comm_size(col_comm, &col_nproc);

  int cnt[row_nproc][col_nproc];
  for (int i = 0; i < row_nproc; ++i) {
    for (int j = 0; j < col_nproc; ++j) {
      cnt[i][j] = 0;
    }
  }

  if (0 == nid) {
    FILE *fp;
    double v;
    size_t i, j, ij;

    int x, y, ex;
    long nnz_pos;

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

    /* send parameters to all nodes */
    for (int id = 1; id < nproc; ++id) {
      MPI_Send(&N,   1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&a,   1, MPI_DOUBLE,   id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nF,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nU,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nI,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
    }

    nnz_pos = ftell(fp);

    for (ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      v = parse_double(fp); /* dummy parse */

      x = get_idx(i, nU, p);
      y = get_idx(j, nI, row_nproc);

      cnt[x][y]++;
    }

    for (i = 0; i < p; ++i) {
      for (j = 0; j < row_nproc; ++j) {
        if ((i == 0) && (j == 0))  continue;
        MPI_Send(&cnt[i][j], 1, MPI_INT, i*p+j, TAG, MPI_COMM_WORLD);
      }
    }

    A = csr_matrix_init(cnt[0][0]);

    rewind(fp);
    fseek(fp, nnz_pos, SEEK_SET);

    for (ij = 0, ex = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      v = parse_double(fp);

      x = get_idx(i, nU, p);
      y = get_idx(j, nI, row_nproc);

      if ((0 == x) && (0 == y)) {
        A->row[ex] = i;
        A->col[ex] = j;
        A->val[ex] = v;
        ex++;
      } else {
        MPI_Send(&i, 1, MPI_UNSIGNED, x*p+y, TAG, MPI_COMM_WORLD);
        MPI_Send(&j, 1, MPI_UNSIGNED, x*p+y, TAG, MPI_COMM_WORLD);
        MPI_Send(&v, 1, MPI_DOUBLE,   x*p+y, TAG, MPI_COMM_WORLD);
      }
    }

    nnz = cnt[0][0];

    if (0 != fclose(fp)) {
      die("main: unable to flush file stream\n");
    }

  } else { /* 0 != nid */
    MPI_Status status;

    MPI_Recv(&N,   1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&a,   1, MPI_DOUBLE,   0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nF,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nU,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nI,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nnz, 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);

    A = csr_matrix_init(nnz);
    
    for (int i = 0; i < nnz; ++i) {
      MPI_Recv(&A->row[i], 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&A->col[i], 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&A->val[i], 1, MPI_DOUBLE,   0, TAG, MPI_COMM_WORLD, &status);
    }
  }

  low_L = col_nid * nU / row_nproc;
  low_R = row_nid * nI / col_nproc;
  high_L = (col_nid + 1) * nU / row_nproc;
  high_R = (row_nid + 1) * nI / col_nproc;

#if 0
  printf("{rank = %d} -> {\n", nid);
  for (int ij = 0; ij < nnz; ++ij) {
    printf("\t(%u, %u, %lf)\n", A->row[ij]-low_L, A->col[ij]-low_R, A->val[ij]);
  }
  printf("}\n");
#endif

  L  = matrix_init(high_L-low_L, nF);
  Lt = matrix_init(high_L-low_L, nF);
  R  = matrix_init(high_R-low_R, nF);
  Rt = matrix_init(high_R-low_R, nF);

  random_fill_L();
  random_fill_R();
  
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

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Finalize();
  return 0;
}
