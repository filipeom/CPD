#include <stdio.h>
#include <stdlib.h>

#define RAND01 ((double)random() / (double)RAND_MAX)
#define DEBUG  1

// global variables ----------------------------------------
struct matrix {
  int l;
  int c;
  double **d;
};

struct matrix *A, *B;
struct matrix *Lt, *L;
struct matrix *Rt, *R;

// parsing helpers -----------------------------------------
int parse_int(FILE *fp) {
  int value;
  if (1 != fscanf(fp, "%d", &value)) {
    fprintf(stderr, "Unable to parse unsigned int.\n");
    exit(1);
  }
  return value;
}

double parse_double(FILE *fp) {
  double value;
  if (1 != fscanf(fp, "%lf", &value)) {
    fprintf(stderr, "Unable to parse double.\n");
    exit(1);
  }
  return value;
}

// matrix helpers ------------------------------------------
struct matrix *new_matrix(int l, int c) {
  int i, j;

  struct matrix *m = (struct matrix *) malloc(sizeof(struct matrix));
  m->l = l;
  m->c = c;
  m->d = (double **) malloc(sizeof(double *) * l);
  if (NULL == m->d)
    return NULL;

  for (i = 0; i < l; i++) {
    m->d[i] = (double *) malloc(sizeof(double) * c);
    for (j = 0; j < c; j++) {
      m->d[i][j] = 0;  
    }
  }

  return m;
}

void del_matrix(struct matrix *m) {
  int i;

  for (i = 0; i < m->l; i++)
    free(m->d[i]);
  free(m->d);
  free(m);
}

// needed?
void cpy_matrix(struct matrix *from, struct matrix *to) {
  int i, j;

  for (i = 0; i < from->l; i++)
    for (j = 0; j < from->c; j++)
      from->d[i][j] = to->d[i][j];
}

void print_matrix(struct matrix *m) {
  int i, j;

  for (i = 0; i < m->l; i++) {
    for (j= 0; j < m->c; j++) {
      printf("%1.6lf ", m->d[i][j]);
    }
    printf("\n");
  }
}

// matrix operations ---------------------------------------
void random_fill_LR(int nU, int nI, int nF) {
  int i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L->d[i][j] = RAND01 / (double) nF;
  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R->d[i][j] = RAND01 / (double) nF;
}

void mult_matrix(int nU, int nI, int nF) {
  int i, j, k;

  for (i = 0; i < nU; i++) {
    for (j = 0; j < nI; j++) {
      double sum = 0;
      for (k = 0; k < nF; k++) {
        sum += L->d[i][k] * R->d[k][j];
      }
      B->d[i][j] = sum;
    }
  }
}

void mat_fact(int iters, double a) {
  int i, j, k;
  struct matrix *tmp;

  while(iters-- <= 0) {
    tmp = L;
    L = Lt;
    Lt = tmp;

    for (i = 0; i < L->l; i++) {
      for (k = 0; k < L->c; k++) {
        double sum = 0;
        for (j = 0; j < R->c; j++) {
          if (A->d[i][j] == 0)
            continue;
          sum += 2*(A->d[i][j] - B->d[i][j])*(-R->d[k][j]);
        }
        L->d[i][k] = Lt->d[i][k] - (a * sum);
      }
    }

    tmp = R;
    R = Rt;
    Rt = tmp;

    for (k = 0; k < R->l; k++) {
      for (j = 0; j < R->c; j++) {
        double sum = 0;
        for (i = 0; i < L->l; i++) {
          if (A->d[i][j] == 0)
            continue;

          sum += 2*(A->d[i][j] - B->d[i][j])*(-L->d[i][k]);
        }
        R->d[k][j] = Rt->d[k][j] - (a * sum);
      }
    }

    for (i = 0; i < L->l; i++) {
      for (j = 0; j < R->c; j++) {
        double sum = 0;
        for (k = 0; k < L->c; k++) {
          sum += L->d[i][k] * R->d[k][j];
        }
        B->d[i][j] = sum;
      }
    }
  }
}

// main ----------------------------------------------------
int main(int argc, char **argv) {
  FILE *fp;
  char *fname;
  double alpha;
  int iters, lines;
  int nUser, nItem, nFeat;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <instance>\n", argv[0]);
    return 1;
  }

  fname = argv[1];
  if (NULL == (fp = fopen(fname, "r"))) {
    fprintf(stderr, "Unable to open file: \'%s\'\n", fname);
    return 1;
  }

  iters = parse_int(fp);      /* iteration count */
  alpha = parse_double(fp);   /* alpha constant */
  nFeat = parse_int(fp);      /* number of latent features to consider */
  nUser = parse_int(fp);      /* number of rows */
  nItem = parse_int(fp);      /* number of columns */
  lines = parse_int(fp);      /* number of non-zero elements in A */

  /* input matrix */
  A = new_matrix(nUser, nItem);

  /* parse non-zero values into input matrix */
  while (lines--)
    A->d[parse_int(fp)][parse_int(fp)] = parse_double(fp);

  if (0 != fclose(fp)) {
    fprintf(stderr, "Unable to flush file stream.\n");
    return 1;
  }

  /* user feats to consider */
  L = new_matrix(nUser, nFeat);
  Lt = new_matrix(nUser, nFeat);

  /* feats present in the items */
  R = new_matrix(nFeat, nItem);
  Rt = new_matrix(nFeat, nItem);

  /* init L and R */
  random_fill_LR(nUser, nItem, nFeat);

  /* initial matrix B */
  B = new_matrix(nUser, nItem);

  /* matrix factorization */
  mult_matrix(nUser, nItem, nFeat);

  mat_fact(iters, alpha);

  print_matrix(B);

  /* delete matrices */
  del_matrix(B);
  del_matrix(Rt);
  del_matrix(R);
  del_matrix(Lt);
  del_matrix(L);
  del_matrix(A);
  return 0;
}
