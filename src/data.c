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

void add_to_square(Square sq, float val) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < sq.size; i++)
        for (int j = 0; j < sq.size; j++)
            sq.mat[i][j] += val;
}

float sum_square(Square sq) {
    float sum = 0.0;

    for (int i = 0; i < sq.size; i++)
        for (int j = 0; j < sq.size; j++)
            sum += sq.mat[i][j];

    return sum;
}

void init_matrix(float** m, float val, int h, int w) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            m[i][j] = val;
}

void init_vector(float* m, float val, int h) {
    for (int i = 0; i < h; i++)
        m[i] = val;
}

int argmax_vector(float* m, int h) {
    float max = -INFINITY;
    int idx = -1;
    for (int i = 0; i < h; i++)
        if (m[i] > max) {
            max = m[i];
            idx = i;
        }
    return idx;
}

float random_range(float min, float max) {
    return min + rand() / (float) RAND_MAX * (max - min);
}

// Initialize matrix with weight in range [-0.5,0.5]
void random_init_matrix(float** m, int h, int w) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            m[i][j] = random_range(-0.2, 0.2);
        }
}

// Initialize vector with weight in range [-0.5,0.5]
void random_init_vector(float* m, int h) {
    for (int i = 0; i < h; i++)
        m[i] = (float)(rand() - RAND_MAX/2)/(float)RAND_MAX;
}

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
            for(int k = 0; k < b; k++) { 
                R[i][j] += m1[k] * m2[k];
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
        for (int j = 0; j < w; j++) {
            printf("%.2f ", m[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Creates data tensor of shape [batch_size, features]
Data1D* CreateData1D(int features, int batch_size) {
    Data1D* d = (Data1D*) malloc(sizeof(Data1D));
    d->n = features;
    d->b = batch_size;
    d->mat = fmatrix_allocate_2d(batch_size, features);
    d->type = D1D;
    return d;
}

Data1D* CopyData1D(Data1D* data) {
    Data1D* d = (Data1D*) malloc(sizeof(Data1D));
    d->n = data->n;
    d->b = data->b;
    d->mat = fmatrix_allocate_2d(d->b, d->n);
    
    for (int i = 0; i < d->b; i++)
        for (int j = 0; j < d->n; j++)
            d->mat[i][j] = data->mat[i][j];

    d->type = D1D;
    return d;
}

void DestroyData1D(Data1D* d) {
    free_fmatrix_2d(d->mat);
    d->mat = NULL;
    free(d);
}

// Creates data tensor of shape [batch_size, channels, size, size]
Data2D* CreateData2D(int size, int batch_size, int channels) {
    Data2D* d = (Data2D*) malloc(sizeof(Data2D));
    d->size = size;
    d->b = batch_size;
    d->c = channels;
    d->data = square_allocate_2d(batch_size, channels);
    d->type = D2D;
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < channels; j++)
            d->data[i][j] = CreateSquareMatrix(size);
    return d;
}

Data2D* CopyData2D(Data2D* data) {
    Data2D* d = (Data2D*) malloc(sizeof(Data2D));
    d->size = data->size;
    d->b = data->b;
    d->c = data->c;
    d->data = square_allocate_2d(d->b, d->c);
    d->type = D2D;
    for (int i = 0; i < d->b; i++)
        for (int j = 0; j < d->c; j++)
            d->data[i][j] = CopySquareMatrix(data->data[i][j]);
    return d;
}

// Creates data tensor of shape [batch_size, channels, size, size]
Data2D* CreateData2DZeros(int size, int batch_size, int channels) {
    Data2D* d = (Data2D*) malloc(sizeof(Data2D));
    d->size = size;
    d->b = batch_size;
    d->c = channels;
    d->data = square_allocate_2d(batch_size, channels);
    d->type = D2D;
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < channels; j++)
            d->data[i][j] = CreateZerosMatrix(size);
    return d;
}

void RandomInitData2D(Data2D* d) {
    for (int i = 0; i < d->b; i++)
        for (int j = 0; j < d->c; j++)
            random_init_matrix(d->data[i][j].mat, d->size, d->size);
}

void ClearData2D(Data2D* d) {
    for (int i = 0; i < d->b; i++)
        for (int j = 0; j < d->c; j++)
            init_square(d->data[i][j], 0.0);
}

void DestroyData2D(Data2D* d) {
    if (d->data == NULL) {
        return;
    }
    for (int i = 0; i < d->b; i++)
        for (int j = 0; j < d->c; j++)
            DestroySquareMatrix(d->data[i][j]);
    free(d->data[0]);
    free(d->data);
    d->data = NULL;
    free(d);
}

void print_data1d(Data1D* d) {
    printf("Data1D of shape: [%d, %d]\n", d->b, d->n);
    print_matrix(d->mat, d->b, d->n);
}

void print_2d_square_array(Square** data, int vsize, int wsize, int size) {
    for (int i=0; i<vsize; i++) {
        for (int k=0; k<size; k++) {
            for (int j=0; j<wsize; j++) {
                for (int l=0; l<size; l++) {
                    printf("%.4f ", data[i][j].mat[k][l]);
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_data2d(Data2D* d) {
    printf("Data2D of shape: [%d, %d, %d, %d]\n", d->b, d->c, d->size, d->size);
    print_2d_square_array(d->data, d->b, d->c, d->size);
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
