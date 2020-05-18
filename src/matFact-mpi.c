#include <mpi.h>
#include <omp.h>
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

/* globals */
static int rr   = 0;
static int rc   = 0;
static int np   = 0;       /* total number of nodes */
static int gid  = 0;       /* global node id */
static int rid  = 0;       /* row id */
static int cid  = 0;       /* column id */
static uint N   = 0;       /* iteration count */
static uint nF  = 0;       /* number of feats */
static uint nU  = 0;       /* number of users */
static uint nI  = 0;       /* number of items */
static uint nnz = 0;       /* number of !zero */
static uint mU  = 0;       /* node value of nU */
static uint mI  = 0;       /* node value of nI */
static uint sL  = 0;
static uint sR  = 0;
static uint eL  = 0;
static uint eR  = 0;
static double a = 0;       /* alpha constant */
static double *Lt = NULL;  /* users-feats matrix (prev it) */ 
static double *L  = NULL;  /* users-feats matrix (curr it) */
static double *Rt = NULL;  /* feats-items matrix (prev it) */
static double *R  = NULL;  /* feats-items matrix (curr it) */
static char *argv0   = NULL;
static struct csr *A = NULL;
static MPI_Comm rcomm;
static MPI_Comm ccomm;

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

void
usage(void) {
  die("usage:\n\t%s <input-file>\n", argv0);
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
csr_matrix_init(uint nnz, uint rsz)
{
  struct csr *matrix;

  matrix = (struct csr *)  xmalloc(sizeof(struct csr));

  matrix->row = (uint *)   xmalloc(sizeof(uint) * (rsz + 1));
  matrix->col = (uint *)   xmalloc(sizeof(uint) * nnz);
  matrix->val = (double *) xmalloc(sizeof(double) * nnz);

  memset(matrix->row, '\x00', sizeof(uint) * (rsz + 1));
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
random_fill_L()
{
  size_t i, j;

  srand(0);

  for (i = 0; i < sL; ++i) {
    for (j = 0; j < nF; ++j) {
      rand();
    }
  }
  for (i = 0; i < mU; ++i) {
    for (j = 0; j < nF; ++j) {
      L[i * nF + j] = RAND01 / (double) nF;
    }
  }
  for (i = eL; i < nU; ++i) {
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
    for (j = 0; j < sR; ++j) {
      rand();
    }
    for (j = 0; j < mI; ++j) {
      R[j * nF + i] = RAND01 / (double) nF;
    }
    for (j = eR; j < nI; ++j) {
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
  MPI_Status status;
  omp_lock_t *col_mutex;

  col_mutex = (omp_lock_t *) malloc(sizeof(omp_lock_t) * mI);
  for (size_t j = 0; j < mI; ++j)
    omp_init_lock(&col_mutex[j]);

  my_idx = (uint *)   xmalloc(sizeof(uint)   * mU);
  my_max = (double *) xmalloc(sizeof(double) * mU);

  #pragma omp parallel
  {
    size_t i, j, k, jx;
    double tmp;
    double *m1, *mt1;
    double *m2, *mt2;

    #pragma omp single
    {
      memcpy(Lt, L, sizeof(double) * mU * nF);
      memcpy(Rt, R, sizeof(double) * mI * nF);
    }

    /* factorization of L and R */
    for (size_t it = N; it--; ) {
      #pragma omp single 
      {
        if (rid) memset(L, '\x00', sizeof(double) * mU * nF);
        if (cid) memset(R, '\x00', sizeof(double) * mI * nF);
      }

      #pragma omp for schedule(guided)
      for (i = 0; i < mU; ++i) {
        m1 = &L[i * nF], mt1 = &Lt[i * nF];
        for (jx = A->row[i]; jx < A->row[i + 1]; ++jx) {
          j = A->col[jx] - sR;
          m2 = &R[j * nF], mt2 = &Rt[j * nF];
          tmp = a * 2 * (A->val[jx] - dot_prod(mt1, mt2, nF));
          omp_set_lock(&col_mutex[j]);
          for (k = 0; k < nF; ++k) {
            m1[k] += tmp * mt2[k];
            m2[k] += tmp * mt1[k];
          }
          omp_unset_lock(&col_mutex[j]);
        }
      }
      
      #pragma omp single
      {
        MPI_Barrier(rcomm);
        MPI_Allreduce(L, Lt, mU * nF, MPI_DOUBLE, MPI_SUM, rcomm);

        MPI_Barrier(ccomm);
        MPI_Allreduce(R, Rt, mI * nF, MPI_DOUBLE, MPI_SUM, ccomm);

        memcpy(L, Lt, sizeof(double) * mU * nF);
        memcpy(R, Rt, sizeof(double) * mI * nF);
      }
      #pragma omp barrier
    } /* end while */

    #pragma omp for schedule(static) nowait
    for (i = 0; i < mU; ++i) {
      my_max[i] = 0;
      jx = A->row[i];
      mt1 = &Lt[i * nF];
      for (j = 0, mt2 = &Rt[j * nF]; j < mI; ++j, mt2 += nF) {
        if ((A->row[i + 1] - A->row[i]) &&
            (A->row[i + 1] > jx) &&
            ((A->col[jx] - sR) == j)) {
          jx++;
        } else {
          tmp = dot_prod(mt1, mt2, nF);
          if (tmp > my_max[i]) {
            my_max[i] = tmp;
            my_idx[i] = j + sR;
          }
        }
      }
    }
  }

  /* row roots update maxes in their rows */
  if (0 == rid) {
    uint   *aux_idx = (uint *)   xmalloc(sizeof(uint)   * mU);
    double *aux_max = (double *) xmalloc(sizeof(double) * mU);

    for (size_t i = 1; i < rc; ++i) {
      MPI_Recv(aux_idx, mU, MPI_UNSIGNED, i, TAG, rcomm, &status);
      MPI_Recv(aux_max, mU, MPI_DOUBLE,   i, TAG, rcomm, &status);
      for (size_t j = 0; j < mU; ++j) {
        if (aux_max[j] > my_max[j]) {
          my_max[j] = aux_max[j];
          my_idx[j] = aux_idx[j];
        }
      }
    }
    free(aux_max);
    free(aux_idx);
  } else { /* 0 != rid */
    MPI_Send(my_idx, mU, MPI_UNSIGNED, 0, TAG, rcomm);
    MPI_Send(my_max, mU, MPI_DOUBLE,   0, TAG, rcomm);
  }

  /* global root node outputs result */
  if (0 == gid) {
    int size;
    best = (uint *) xmalloc(sizeof(uint) * ceil(nU / rr));

    for (size_t i = 0; i < mU; ++i) {
      printf("%u\n", my_idx[i]);
    }

    for (size_t i = 1; i < rr; ++i) {
      MPI_Recv(&size, 1,    MPI_INT,      i * rc, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(best,  size, MPI_UNSIGNED, i * rc, TAG, MPI_COMM_WORLD, &status);
      for (size_t j = 0; j < size; ++j) {
        printf("%u\n", best[j]);
      }
    }

    free(best);
  } else if ((0 == rid) && (0 != cid)) {
    MPI_Send(&mU, 1,     MPI_INT,      0, TAG, MPI_COMM_WORLD);
    MPI_Send(my_idx, mU, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD);
  }

  for (size_t j = 0; j < mI; ++j)
    omp_destroy_lock(&col_mutex[j]);

  free(col_mutex);
  free(my_idx);
  free(my_max);
}

int
get_idx(uint v, uint n, int p)
{
  for (int id = 1; id <= p; ++id)
    if ((v >= ((id - 1) * n / p)) &&
        (v <  (id * n / p)))
      return id-1;
  return 0;
}

int
main(int argc, char* argv[])
{
  FILE *fp;
  int color;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &gid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (0 == gid) {
    if (2 != argc) {
      argv0 = argv[0];
      usage();
    }

    if (NULL == (fp = fopen(argv[1], "r"))) {
      die("main: unable to open file \'%s\'\n", argv[1]);
    }

    N   = parse_uint(fp);
    a   = parse_double(fp);
    nF  = parse_uint(fp);
    nU  = parse_uint(fp);
    nI  = parse_uint(fp);
    nnz = parse_uint(fp);

    for (int id = 1; id < np; ++id) {
      MPI_Send(&N,   1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&a,   1, MPI_DOUBLE,   id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nF,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nU,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
      MPI_Send(&nI,  1, MPI_UNSIGNED, id, TAG, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&N,   1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&a,   1, MPI_DOUBLE,   0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nF,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nU,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&nI,  1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
  }

  int rnU = sqrt(nU), rnI = sqrt(nI);
  if (rnU >= nI) { rr = np, rc = 1; }
  else if (rnI >= nU) { rr = 1, rc = np; }
  else {
    double dnp = sqrt(np);
    int inp = sqrt(np);
    if (dnp == inp) { rr = rc = inp; } /* prefer square matrices */
    else {
      /* FIXME: why bruteforce this calculation? ... */
      /* this just gives a 1D grid ... */
      int ie, je;
      if (nU > nI) ie = np, je = np / 2;
      else ie = np / 2, je = np;
      for (int i = 1; i <= ie; ++i) 
        for (int j = 1; j <= je; ++j)
          if (i * j == np) {
            rr = i, rc = j;
          }
    }
  }

  printf("rr=%d, rc=%d, np=%d\n", rr, rc, np);
  die("");

  color = floor(gid / rc);
  MPI_Comm_split(MPI_COMM_WORLD, color, gid, &rcomm);
  MPI_Comm_rank(rcomm, &rid);

  color = (gid % rc) + TAG;
  MPI_Comm_split(MPI_COMM_WORLD, color, gid, &ccomm);
  MPI_Comm_rank(ccomm, &cid);

  sL = cid * nU / rr;
  sR = rid * nI / rc;
  eL = (cid + 1) * nU / rr;
  eR = (rid + 1) * nI / rc;
  mU = eL - sL;
  mI = eR - sR;

  if (0 == gid) {
    int x;
    int y;
    int ex;
    double v;
    size_t i;
    size_t j;
    size_t ij;
    long int nnz_pos;
    int cnt[rr][rc];

    for (i = 0; i < rc; ++i)
      for (j = 0; j < rr; ++j)
        cnt[i][j] = 0;

    nnz_pos = ftell(fp);

    /* count nnz per node */
    for (ij = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      v = parse_double(fp); /* dummy parse */
      x = get_idx(i, nU, rr);
      y = get_idx(j, nI, rc);
      cnt[x][y]++;
    }
    
    for (i = 0; i < rr; ++i) {
      for (j = 0; j < rc; ++j) {
        if ((0 == i) && (0 == j))  continue;
        MPI_Send(&cnt[i][j], 1, MPI_INT, i * rc + j, TAG, MPI_COMM_WORLD);
      }
    }

    A = csr_matrix_init(cnt[0][0], mU);

    rewind(fp);
    fseek(fp, nnz_pos, SEEK_SET);

    for (ij = 0, ex = 0; ij < nnz; ++ij) {
      i = parse_uint(fp);
      j = parse_uint(fp);
      v = parse_double(fp);
      x = get_idx(i, nU, rr);
      y = get_idx(j, nI, rc);
      if ((0 == x) && (0 == y)) {
        A->row[(i-sL)+1]++;
        A->col[ex] = j;
        A->val[ex] = v;
        ex++;
      } else {
        MPI_Send(&i, 1, MPI_UNSIGNED, x * rc + y, TAG, MPI_COMM_WORLD);
        MPI_Send(&j, 1, MPI_UNSIGNED, x * rc + y, TAG, MPI_COMM_WORLD);
        MPI_Send(&v, 1, MPI_DOUBLE,   x * rc + y, TAG, MPI_COMM_WORLD);
      }
    }

    for (i = 1; i <= mU; ++i)
      A->row[i] += A->row[i-1];

    nnz = cnt[0][0];

    if (0 != fclose(fp))
      die("main: unable to flush file stream\n");

  } else { /* 0 != gid */
    size_t ij;
    MPI_Recv(&nnz, 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);

    A = csr_matrix_init(nnz, mU);
    
    for (ij = 0; ij < nnz; ++ij) {
      uint i;
      MPI_Recv(&i, 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&A->col[ij], 1, MPI_UNSIGNED, 0, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&A->val[ij], 1, MPI_DOUBLE,   0, TAG, MPI_COMM_WORLD, &status);
      A->row[(i-sL)+1]++;
    }

    for (size_t i = 1; i <= mU; ++i)
      A->row[i] += A->row[i-1];
  }

  L  = matrix_init(mU, nF);
  Lt = matrix_init(mU, nF);
  R  = matrix_init(mI, nF);
  Rt = matrix_init(mI, nF);

  random_fill_L();
  random_fill_R();

  double secs;
  secs = - MPI_Wtime();
  solve();
  secs += MPI_Wtime();

  // Redirect stdout to file and get time on stderr
  if (0 == gid) fprintf(stderr, "Time = %12.6f sec\n", secs);

  matrix_destroy(&Rt);
  matrix_destroy(&R);
  matrix_destroy(&Lt);
  matrix_destroy(&L);
  csr_matrix_destroy(&A);
  MPI_Comm_free(&rcomm);
  MPI_Comm_free(&ccomm);
  MPI_Finalize();
  return 0;
}
