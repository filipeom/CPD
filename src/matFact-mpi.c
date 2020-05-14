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

static uint my_nU  = 0;
static uint my_nI  = 0;

static uint low_L  = 0;
static uint low_R  = 0;
static uint high_L = 0;
static uint high_R = 0;

static double *Lt = NULL;  /* users-feats matrix (prev it) */ 
static double *L  = NULL;  /* users-feats matrix (curr it) */
static double *Rt = NULL;  /* feats-items matrix (prev it) */
static double *R  = NULL;  /* feats-items matrix (curr it) */
static double *B  = NULL;

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
  for (i = 0; i < my_nU; ++i) {
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
    for (j = 0; j < my_nI; ++j) {
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
  uint   *best;
  uint   *my_idx;
  double *my_max;
  size_t i, j, k, ij;
  double tmp, max_nU;
  double *m1, *mt1;
  double *m2, *mt2;

  for (i = 0, max_nU = 0; i < row_nproc; ++i) {
    int low = i * nU / row_nproc;
    int high = (i + 1) * nU / row_nproc;
    max_nU = ((high - low) > max_nU) ? (high - low) : max_nU;
  }

  if (0 == nid) {
    best = (uint *) xmalloc(sizeof(uint) * max_nU);
  }

  my_idx = (uint *)   xmalloc(sizeof(uint)   * my_nU);
  my_max = (double *) xmalloc(sizeof(double) * my_nU);

  memcpy(Lt, L, sizeof(double) * my_nU * nF);
  memcpy(Rt, R, sizeof(double) * my_nI * nF);

  while (N--) {
    if (row_nid) memset(L, '\x00', sizeof(double) * my_nU * nF);
    if (col_nid) memset(R, '\x00', sizeof(double) * my_nI * nF);

    for (ij = 0; ij < nnz; ++ij) {
      i = A->row[ij] - low_L, j = A->col[ij] - low_R;
      m1 = &L[i * nF], mt1 = &Lt[i * nF];
      m2 = &R[j * nF], mt2 = &Rt[j * nF];
      tmp = a * 2 * (A->val[ij] - dot_prod(mt1, mt2, nF));
      for (k = 0; k < nF; ++k) {
        m1[k] += tmp * mt2[k];
        m2[k] += tmp * mt1[k];
      }
    }
    
    memset(Lt, '\x00', sizeof(double) * my_nU * nF);

    MPI_Allreduce(
        L,
        Lt,
        my_nU * nF,
        MPI_DOUBLE,
        MPI_SUM,
        row_comm
    );

    MPI_Barrier(row_comm);

    memset(Rt, '\x00', sizeof(double) * my_nI * nF);

    MPI_Allreduce(
        R,
        Rt,
        my_nI * nF,
        MPI_DOUBLE,
        MPI_SUM,
        col_comm
    );

    MPI_Barrier(col_comm);

    memcpy(L, Lt, sizeof(double) * my_nU * nF);
    memcpy(R, Rt, sizeof(double) * my_nI * nF);
  } /* end while */

  for (i = 0; i < my_nU; ++i) {
    for (j = 0; j < my_nI; ++j) {
      B[i * my_nI + j] = dot_prod(&Lt[i * nF], &Rt[j * nF], nF);
    }
  }

  for (ij = 0; ij < nnz; ++ij) {
    i = A->row[ij] - low_L, j = A->col[ij] - low_R;
    B[i * my_nI + j] = 0;
  }

  for (i = 0; i < my_nU; ++i) {
    my_max[i] = 0;
    for (j = 0; j < my_nI; ++j) {
      tmp = B[i * my_nI + j];
      if (tmp > my_max[i]) {
        my_max[i] = tmp;
        my_idx[i] = j + low_R;
      }
    }
  }

  MPI_Barrier(row_comm);

  if (row_nid) {
    MPI_Send(my_idx, my_nU, MPI_UNSIGNED, 0, TAG, row_comm);
    MPI_Send(my_max, my_nU, MPI_DOUBLE,   0, TAG, row_comm);
  } else {
    MPI_Status status;
    uint   aux_idx[my_nU];
    double aux_max[my_nU];
    for (i = 1; i < row_nproc; ++i) {
      MPI_Recv(aux_idx, my_nU, MPI_UNSIGNED, i, TAG, row_comm, &status);
      MPI_Recv(aux_max, my_nU, MPI_DOUBLE,   i, TAG, row_comm, &status);
      for (j = 0; j < my_nU; ++j) {
        if (aux_max[j] > my_max[j]) {
          my_max[j] = aux_max[j];
          my_idx[j] = aux_idx[j];
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (0 == nid) {
    MPI_Status status;
    for (i = 0; i < my_nU; ++i) { printf("%u\n", my_idx[i]); }
#if 0
    for (i = 1; i < col_nproc; ++i) {
      tmp = ((i + 1) * nU / row_nproc) - (i * nU / row_nproc);
      MPI_Recv(best, tmp, MPI_UNSIGNED, i, TAG, row_comm, &status);
      for (j = 0; j < tmp; ++j) { printf("%u\n", best[j]); }
    }
#endif
    free(best);
  } else {
    if ((0 == row_nid) && (0 != nid)) {
      //MPI_Send(my_idx, my_nU, MPI_UNSIGNED, 0, TAG, row_comm);
      for (i = 0; i < my_nU; ++i) { printf("%u\n", my_idx[i]); }
    }
  }

  free(my_idx);
  free(my_max);
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

  my_nU = high_L - low_L;
  my_nI = high_R - low_R;

#if 0
  printf("{rank = %d} -> {\n", nid);
  for (int ij = 0; ij < nnz; ++ij) {
    printf("\t(%u, %u, %lf)\n", A->row[ij]-low_L, A->col[ij]-low_R, A->val[ij]);
  }
  printf("}\n");
#endif

  L  = matrix_init(my_nU, nF);
  Lt = matrix_init(my_nU, nF);
  R  = matrix_init(my_nI, nF);
  Rt = matrix_init(my_nI, nF);
  B  = matrix_init(my_nU, my_nI);

  random_fill_L();
  random_fill_R();

  
  double secs;
  secs = - MPI_Wtime();
  solve();
  secs += MPI_Wtime();
  
  // Redirect stdout to file and get time on stderr
  if (0 == nid) fprintf(stderr, "Time = %12.6f sec\n", secs);

  matrix_destroy(&B);
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
