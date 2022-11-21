#include "lib.h"

// Basic convolution, stride of 1, no padding (or included in X), only squares
void conv_forward(Square X, Square W, Square Y, float (*activation)(float)) {
    #pragma omp parallel for
    for(int i=0; i < Y.size; i++)
        for(int j=0; j < Y.size; j++) {
            float sum = 0.0;
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) 
                    sum += W.mat[k][l] * X.mat[k+i][l+j];
            Y.mat[i][j] = activation(sum);
        }
}

// Backward convolutional pass, computing in dX and dW the gradient of the next
// layer Y w.r.t X and W 
BackwardConvResult conv_backward(Square dY, Square X, Square W) {
    BackwardConvResult r;
    r.dX = CreateZerosMatrix(X.size);
    r.dW = CreateZerosMatrix(W.size);

    #pragma omp parallel for
    for(int i=0; i < dY.size; i++)
        for(int j=0; j < dY.size; j++) {
            float dYElem = dY.mat[i][j];
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) {
                    r.dW.mat[k][l] += X.mat[k+i][l+j] * dYElem;
                    r.dX.mat[k+i][l+j] += W.mat[k][l] * dYElem;
                }
        }

    return r;
}

/*
    @param layer.w: [output_features, input_features] 
    @param inputs: [batch_size, input_features]
    @returns outputs: [batch_size, output_features] */
Data1D linear_forward(LinearLayer layer, Data1D inputs) {
    assert(layer.in == inputs.n && "Invalid size of input data for linear layer");
    
    Data1D outputs = CreateData1D(layer.out, inputs.b);
    matrix_mul_2d_T2(inputs.mat, layer.w, outputs.mat, inputs.b, layer.in, layer.out);
    return outputs;
}

// @param dY: [batch_size, output_features]
// @param  X: [batch_size, input_features]
// @param layer.w: [output_features, input_features]
// @returns dX: [batch_size, input_features]
// @returns layer.dW: [output_features, input_features]
Data1D linear_backward(Data1D dY, Data1D X, LinearLayer layer) {
    assert(layer.dW != NULL && "This linear layer wasn't initialized with the with_gradient flag");
    assert(dY.n == layer.out && "The gradient from the next layer has not the same number of features than this layer");
    assert(dY.b == X.b && "The gradient from the next layer is not computed for the same number of batch than the input of this layer");

    Data1D dX = CreateData1D(X.n, X.b);
    init_matrix(dX.mat, 0.0, dX.b, dX.n);
    init_matrix(layer.dW, 0.0, layer.out, layer.in);
    // dX = dY * layer.w
    matrix_mul_2d(dY.mat, layer.w, dX.mat, dX.b, layer.out, dX.n);
    // dW = dY.T * X
    matrix_mul_2d_T1(dY.mat, X.mat, layer.dW, layer.out, X.b, layer.in);
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

Data1D CreateData1D(int features, int batch_size) {
    Data1D d;
    d.n = features;
    d.b = batch_size;
    d.mat = fmatrix_allocate_2d(batch_size, features);
    return d;
}

void DestroyData1D(Data1D d) {
    free_fmatrix_2d(d.mat);
}

Data1D squeeze(Data2D d) {
    Data1D d_ = CreateData1D(d.size*d.size, d.b);

}

LinearLayer CreateLinearLayer(int in_channels, int out_channels, int with_gradient) {
    LinearLayer l;
    l.w = fmatrix_allocate_2d(out_channels, in_channels);
    l.in = in_channels;
    l.out = out_channels;

    l.dW = with_gradient ? fmatrix_allocate_2d(out_channels, in_channels) : NULL;
 
    return l;
}

void DestroyLinearLayer(LinearLayer layer) {
    free_fmatrix_2d(layer.w);
    if (layer.dW != NULL) free_fmatrix_2d(layer.dW);
}

ConvLayer CreateConvLayer(int in_channels, int out_channels, int size) {
    ConvLayer c;
    c.kernels = square_allocate_2d(out_channels, in_channels);
    for (int i; i<in_channels; i++) 
        for (int j; j<out_channels; j++)
            c.kernels[i][j] = CreateSquareMatrix(size);
    c.in = in_channels;
    c.out = out_channels;
    c.size = size;
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

// Initialize matrix with weight in range [0, 1]
void random_init_matrix(float** m, int h, int w) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            m[i][j] = (float)rand()/(float)RAND_MAX;
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
    #pragma omp parallel for shared(M1T,M2,R)
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


// Returns Y size of a convulation of a W of size filter_size
// on an X of size X_size
int get_output_size(int input_size, int filter_size) {
    return input_size - filter_size + 1;
} 