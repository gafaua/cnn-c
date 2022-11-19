#include "lib.h"

// Basic convolution, stride of 1, no padding (or included in X), only squares
void conv_forward(Square X, Square W, Square Y, float (*activation)(float)) {
    #pragma omp parallel for
    for(int i=0; i < Y.shape; i++)
        for(int j=0; j < Y.shape; j++) {
            float sum = 0.0;
            for(int k=0; k < W.shape; k++)
                for(int l=0; l < W.shape; l++) 
                    sum += W.mat[k][l] * X.mat[k+i][l+j];
            Y.mat[i][j] = activation(sum);
        }
}

// Backward convolutional pass, computing in dX and dW the gradient of the next
// layer Y w.r.t X and W 
// TODO MAKE PARALLEL
BackwardPassResult conv_backward(Square dY, Square X, Square W) {
    BackwardPassResult r;
    r.dX = CreateZerosMatrix(X.shape);
    r.dW = CreateZerosMatrix(W.shape);

    for(int i=0; i < dY.shape; i++)
        for(int j=0; j < dY.shape; j++)
            for(int k=0; k < W.shape; k++)
                for(int l=0; l < W.shape; l++) {
                    r.dW.mat[k][l] += X.mat[k+i][l+j] * dY.mat[i][j];
                    r.dX.mat[k+i][l+j] += W.mat[k][l] * dY.mat[i][j];
                }
    
    return r;
}

float ReLU(float val) {
    if (val <= 0.0) return 0.0;
    else return val;
}

float ReLU_backward(float val) {
    if (val <= 0.0) return 0.0;
    else return 1.0;
}

float Identity(float val) {
    return val;
}

LinearLayer CreateLinearLayer(int in_channels, int out_channels) {
    LinearLayer l;
    l.w = fmatrix_allocate_2d(in_channels, out_channels);
    l.in = in_channels;
    l.out = out_channels;
    return l;
}

void DestroyLinearLayer(LinearLayer layer) {
    free_fmatrix_2d(layer.w);
}

ConvLayer CreateConvLayer(int in_channels, int out_channels, int shape) {
    ConvLayer c;
    c.kernels = square_allocate_2d(in_channels, out_channels);
    for (int i; i<in_channels; i++) 
        for (int j; j<out_channels; j++)
            c.kernels[i][j] = CreateSquareMatrix(shape);
    c.in = in_channels;
    c.out = out_channels;
    c.shape = shape;
}

void DestroyConvLayer(ConvLayer c) {
    for (int i; i<c.in; i++) 
        for (int j; j<c.out; j++)
            DestroySquareMatrix(c.kernels[i][j]);
    free(c.kernels[0]);
    free(c.kernels);
}

Square CreateSquareMatrix(int size) {
    Square s;
    s.mat = fmatrix_allocate_2d(size, size);
    s.shape = size;
    return s;
}

Square CreateZerosMatrix(int size) {
    Square s = CreateSquareMatrix(size);
    init_matrix(s.mat, 0.0, size, size);
    return s;
}

Square CopySquareMatrix(Square sq) {
    Square s;
    s.mat = fmatrix_allocate_2d(sq.shape, sq.shape);
    s.shape = sq.shape;

    for(int i=0; i < sq.shape; i++)
        for(int j=0; j < sq.shape; j++) 
            s.mat[i][j] = sq.mat[i][j];

    return s;
}

void DestroySquareMatrix(Square s) {
    free_fmatrix_2d(s.mat);
}

void init_square(Square sq, float val) {
    init_matrix(sq.mat, val, sq.shape, sq.shape);
}

void init_matrix(float** m, float val, int h, int w) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            m[i][j] = val;
}

void print_square(Square s) {
    printf("Matrix of shape %d x %d:\n", s.shape, s.shape);
    print_matrix(s.mat, s.shape, s.shape);
}

void print_matrix(float** m, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            printf("%.2f ", m[i][j]);
        printf("\n");
    }
    printf("\n");
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

//----------------------------------------------------------*/
/* Libere la memoire de la matrice 2d de float              */
/*----------------------------------------------------------*/
void free_fmatrix_2d(float** pmat)
 { 
  free(pmat[0]);
  free(pmat);
 }


// Returns Y shape of a convulation of a W of size filter_size
// on an X of size X_size
int get_output_shape(int input_size, int filter_size) {
    return input_size - filter_size + 1;
} 