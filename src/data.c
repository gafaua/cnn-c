#include "data.h"

Square CreateSquareMatrix(int size) {
    Square s;
    s.mat = fmatrix_allocate_2d(size, size);
    s.size = size;
    return s;
}

Square CreateZerosMatrix(int size) {
    Square s = CreateSquareMatrix(size);
    init_matrix(s.mat, 0.0, size, size);
    return s;
}

Square CopySquareMatrix(Square sq) {
    Square s;
    s.mat = fmatrix_allocate_2d(sq.size, sq.size);
    s.size = sq.size;

    for(int i=0; i < sq.size; i++)
        for(int j=0; j < sq.size; j++) 
            s.mat[i][j] = sq.mat[i][j];

    return s;
}

void DestroySquareMatrix(Square s) {
    free_fmatrix_2d(s.mat);
    s.mat = NULL;
}

void init_square(Square sq, float val) {
    init_matrix(sq.mat, val, sq.size, sq.size);
}

void init_matrix(float** m, float val, int h, int w) {
    #pragma omp parallel for
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            m[i][j] = val;
}

// Initialize matrix with weight in range [-0.5,0.5]
void random_init_matrix(float** m, int h, int w) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            m[i][j] = (float)(rand() - RAND_MAX/2)/(float)RAND_MAX;
        }
}

// TODO test speed, try to transpose M2
// M1: [a, b] x M2: [b, c] -> R: [a, c]
void matrix_mul_2d(float** M1, float** M2, float** R, int a, int b, int c) {
    #pragma omp parallel for shared(M1,M2,R)
    for(int i = 0; i < a; i++) {
        for(int k = 0; k < b; k++) { 
            float* r = R[i];
            float m1 = M1[i][k];
            float* m2 = M2[k];
            for(int j = 0; j < c; j++) {
                r[j] += m1 * m2[j];
	        }
	    }
    }
}

// M1T: [b, a] x M2: [b, c] -> R: [a, c]
void matrix_mul_2d_T1(float** M1T, float** M2, float** R, int a, int b, int c) {
    #pragma omp parallel for shared(M1T,M2,R) collapse(2)
    for(int k = 0; k < b; k++) { 
        for(int i = 0; i < a; i++) {
            float* m2 = M2[k];
            float m1 = M1T[k][i];
            float* r = R[i];
            for(int j = 0; j < c; j++) {
                r[j] += m1 * m2[j];
	        }
	    }
    }
}

// M1: [a, b] x M2T: [c, b] -> R: [a, c]
void matrix_mul_2d_T2(float** M1, float** M2T, float** R, int a, int b, int c) {
    #pragma omp parallel for shared(M1,M2T,R)
    for(int i = 0; i < a; i++) {
        float* m1 = M1[i];
        for(int j = 0; j < c; j++) {
            float* m2 = M2T[j];
            float* r = &R[i][j];
            for(int k = 0; k < b; k++) { 
                *r += m1[k] * m2[k];
	        }
	    }
    }
}

void print_square(Square s) {
    printf("Matrix of size %d x %d:\n", s.size, s.size);
    print_matrix(s.mat, s.size, s.size);
}

void print_matrix(float** m, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            printf("%.2f ", m[i][j]);
        printf("\n");
    }
    printf("\n");
}

void print_data1d(Data1D* d) {
    printf("Data1D of shape: [%d, %d]\n", d->b, d->n);
    print_matrix(d->mat, d->b, d->n);
}

void print_data2d(Data2D* d) {
    printf("Data2D of shape: [%d, %d, %d, %d]\n", d->b, d->c, d->size, d->size);
    for (int i=0; i<d->b; i++) {
        for (int k=0; k<d->size; k++) {
            for (int j=0; j<d->c; j++) {
                for (int l=0; l<d->size; l++) {
                    printf("%.2f ", d->data[i][j].mat[k][l]);
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("\n");
    }
}

/*----------------------------------------------------------*/
/*  Alloue de la memoire pour une matrice 2d de float       */
/*----------------------------------------------------------*/
float** fmatrix_allocate_2d(int vsize,int hsize)
 {
  int i;
  float** matrix;
  float *imptr;

  matrix=(float**)malloc(sizeof(float*)*vsize);
  if (matrix==NULL) printf("probleme d'allocation memoire");

  imptr=(float*)malloc(sizeof(float)*hsize*vsize);
  if (imptr==NULL) printf("probleme d'allocation memoire");
 
  for(i=0;i<vsize;i++,imptr+=hsize) matrix[i]=imptr;
  return matrix;
 }


/*----------------------------------------------------------*/
/*  Alloue de la memoire pour une matrice 2d de Square       */
/*----------------------------------------------------------*/
Square** square_allocate_2d(int vsize,int hsize)
 {
  int i;
  Square** matrix;
  Square* imptr;

  matrix=(Square**)malloc(sizeof(Square*)*vsize);
  if (matrix==NULL) printf("probleme d'allocation memoire");

  imptr=(Square*)malloc(sizeof(Square)*hsize*vsize);
  if (imptr==NULL) printf("probleme d'allocation memoire");
 
  for(i=0;i<vsize;i++,imptr+=hsize) matrix[i]=imptr;
  return matrix;
 }


/*----------------------------------------------------------*/
/*  Alloue de la memoire pour une matrice 1d de Square       */
/*----------------------------------------------------------*/
Square* square_allocate_1d(int size)
 {
  Square* matrix;

  matrix=(Square*)malloc(sizeof(Square)*size);
  if (matrix==NULL) printf("probleme d'allocation memoire");
 
  return matrix;
 }

//----------------------------------------------------------*/
/* Libere la memoire de la matrice 2d de float              */
/*----------------------------------------------------------*/
void free_fmatrix_2d(float** pmat)
 { 
  free(pmat[0]);
  free(pmat);
 }
