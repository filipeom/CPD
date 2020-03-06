#include <stdio.h>
#include <stdlib.h>

#define RAND01 ((double)random() / (double)RAND_MAX)

// global variables ----------------------------------------
double **A, **B;
double **Lt, **L;
double **Rt, **R;

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
double **new_matrix(int l, int c) {
  int i, j;

  double **m = (double **) malloc(sizeof(double *) * l);
  if (NULL == m)
    return NULL;

  for (i = 0; i < l; i++) {
    m[i] = (double *) malloc(sizeof(double) * c);
    for (j = 0; j < c; j++) {
      m[i][j] = 0;  
    }
  }

  return m;
}

void del_matrix(double **m, int l) {
  int i;

  for (i = 0; i < l; i++)
    free(m[i]);
  free(m);
  return;
}

void print_matrix(double **m, int l, int c) {
  int i, j;

  for (i = 0; i < l; i++) {
    for (j= 0; j < c; j++) {
      printf("%1.6lf ", m[i][j]);
    }
    printf("\n");
  }
  return;
}

// matrix operations ---------------------------------------
void random_fill_LR(int nU, int nI, int nF) {
  int i, j;

  srandom(0);

  for (i = 0; i < nU; i++)
    for (j = 0; j < nF; j++)
      L[i][j] = RAND01 / (double) nF;

  for (i = 0; i < nF; i++)
    for (j = 0; j < nI; j++)
      R[i][j] = RAND01 / (double) nF;
  return;
}

void matrix_mult_LR(int nU, int nI, int nF) {
  int i, j, k;

  for (i = 0; i < nU; i++) {
    for (j = 0; j < nI; j++) {
      double sum = 0;
      for (k = 0; k < nF; k++) {
        sum += L[i][k] * R[k][j];
      }
      B[i][j] = sum;
    }
  }
  return;
}

void matrix_fact_B(int nU, int nI, int nF, int n, double a) {
  int i, j, k;
  double **tmp;

  while(--n) {
    tmp = L; L = Lt; Lt = tmp;
    tmp = R; R = Rt; Rt = tmp;

    for (i = 0; i < nU; i++) {
      for (k = 0; k < nF; k++) {
        double sum = 0;
        for (j = 0; j < nI; j++) {
          if (A[i][j])
            sum += 2*(A[i][j] - B[i][j])*(-Rt[k][j]);
        }
        L[i][k] = Lt[i][k] - (a * sum);
      }
    }

    for (k = 0; k < nF; k++) {
      for (j = 0; j < nI; j++) {
        double sum = 0;
        for (i = 0; i < nU; i++) {
          if (A[i][j])
            sum += 2*(A[i][j] - B[i][j])*(-Lt[i][k]);
        }
        R[k][j] = Rt[k][j] - (a * sum);
      }
    }

    matrix_mult_LR(nU, nI, nF);
  }
  return;
}

void recommend(int l, int c) {
  int i, j;

  for (i = 0; i < l; i++) {
    int item = 0;
    double max = -1.00;
    for (j = 0; j < c; j++) {
      if ((B[i][j] > max) && !A[i][j]) {
        max = B[i][j];
        item = j;
      }
    }
    printf("%d\n", item);
  }
  return;
}

// main ----------------------------------------------------
int main(int argc, char **argv) {
  FILE *fp;
  double alpha;
  int N, lines, i, j;
  int numU, numI, numF;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <instance>\n", argv[0]);
    return 1;
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Unable to open file: \'%s\'\n", argv[1]);
    return 1;
  }

  N = parse_int(fp);
  alpha = parse_double(fp);
  numF = parse_int(fp);
  numU = parse_int(fp); numI = parse_int(fp); lines = parse_int(fp);

  A = new_matrix(numU, numI);
  while (lines--) {
    i = parse_int(fp);
    j = parse_int(fp);
    A[i][j] = parse_double(fp);
  }

  if (fclose(fp) != 0) {
    fprintf(stderr, "Unable to flush file stream.\n");
    return 1;
  }

  L  = new_matrix(numU, numF);
  Lt = new_matrix(numU, numF);
  R  = new_matrix(numF, numI);
  Rt = new_matrix(numF, numI);
  B  = new_matrix(numU, numI);

  random_fill_LR(numU, numI, numF);
  matrix_mult_LR(numU, numI, numF);
  matrix_fact_B(numU, numI, numF, N, alpha);
  recommend(numU, numI);

  del_matrix(B, numU);
  del_matrix(Rt, numF); 
  del_matrix(R, numF);
  del_matrix(Lt, numU);
  del_matrix(L, numU);
  del_matrix(A, numU);
  return 0;
}
