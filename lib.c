#include "lib.h"

// Y must be 0 init
void convolution(Square X, Square W, Square Y) {
    #pragma omp parallel for
    for(int i=0; i < Y.size; i++)
        for(int j=0; j < Y.size; j++) {
            float sum = 0.0;
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) 
                    sum += W.mat[k][l] * X.mat[k+i][l+j];
            Y.mat[i][j] += sum;
        }
}

void deconvolution(Square dY, Square X, Square dX, Square W, Square dW) {
    #pragma omp parallel for
    for(int i=0; i < dY.size; i++)
        for(int j=0; j < dY.size; j++) {
            float dYElem = dY.mat[i][j];
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) {
                    dW.mat[k][l] += X.mat[k+i][l+j] * dYElem;
                    dX.mat[k+i][l+j] += W.mat[k][l] * dYElem;
                }
        }
}

// @brief Basic convolution, stride of 1, no padding (or included in X), only squares
// @param layer: Convolutionnal layer, weights of dim [out_channels, in_channels, k_size, k_size]
// @param input: Data2D, dim [batch, in_channels, size, size]
// @returns Data2D, dim [batch, out_channels, size, size]
Data2D conv_forward(ConvLayer* layer, Data2D* input) {
    assert(input->c == layer->in && "The input must have the same number of channels as the input channels of this layer");

    if (layer->dW != NULL) {
        // Save inputs in layer->X
        if (layer->X.data != NULL) {
            DestroyData2D(&layer->X);
        }

        layer->X.b = input->b;
        layer->X.c = input->c;
        layer->X.size = input->size;
        layer->X.data = input->data;
    }

    int output_size = get_output_size(input->size, layer->size);
    Data2D output = CreateData2DZeros(output_size, input->b, layer->out);
    int i,j,k;

    //#pragma omp parallel for
    for (i = 0; i < output.b; i++) {
        Square* in = input->data[i];
        for (j = 0; j < output.c; j++) {
            Square* kernels = layer->w[j];
            Square out = output.data[i][j];
            for (k = 0; k < input->c; k++)
                convolution(in[k], kernels[k], out);
        }
    }

    return output;
}

// TODO TEEEEEEEST
// @param dY: Data2D, dim [batch, out_channels, size, size]
// @param X: Data2D, dim [batch, in_channels, size, size]
// @param layer: Convolutionnal layer, weights of dim [out_channels, in_channels, k_size, k_size]
// @returns dX: Data2D, dim [batch, in_channels, size, size]
// @returns layer.dW: dim [out_channels, in_channels, k_size, k_size]
Data2D conv_backward(ConvLayer* layer, Data2D* dY) {
    assert(layer->dW != NULL && "This convolutional layer wasn't initialized with the with_gradient flag");
    assert(layer->out == dY->c && "The gradient from the next layer doesn't have the same number of channels as the output channels of this convolutional layer");
    assert(layer->in == layer->X.c && "The input tensor given doesn't have the same number of channels as the input channels of this convolutional layer");

    int i,j,k;

    // Clear gradients
    Data2D dX = CreateData2DZeros(layer->X.size, layer->X.b, layer->X.c);
    for (i = 0; i < layer->out; i++)
        for (j = 0; j < layer->in; j++)
            init_square(layer->dW[i][j], 0.0);

    // Compute new gradients
    for (k = 0; k < layer->X.b; k++) {
        Square* dYb = dY->data[k];
        Square* Xb = layer->X.data[k];
        Square* dXb = dX.data[k];
        for (i = 0; i < layer->out; i++) {
            Square* kernels = layer->w[i];
            Square* dWOut = layer->dW[i];
            for (j = 0; j < layer->in; j++) {
                deconvolution(dYb[i], Xb[j], dXb[j], kernels[j], dWOut[j]);
            }
        }
    }
    return dX;
}

/*
    @param layer.w: [output_features, input_features] 
    @param inputs: [batch_size, input_features]
    @returns outputs: [batch_size, output_features] */
Data1D linear_forward(LinearLayer* layer, Data1D* input) {
    assert(layer->in == input->n && "Invalid size of input data for linear layer");
    
    if (layer->dW != NULL) {
        // Save inputs in layer.X
        if (layer->X.mat != NULL) {
            DestroyData1D(&layer->X);
        }

        layer->X.b = input->b;
        layer->X.n = input->n;
        layer->X.mat = input->mat;
    }

    Data1D outputs = CreateData1D(layer->out, input->b);
    matrix_mul_2d_T2(input->mat, layer->w, outputs.mat, input->b, layer->in, layer->out);
    return outputs;
}

// @param dY: [batch_size, output_features]
// @param  X: [batch_size, input_features]
// @param layer.w: [output_features, input_features]
// @returns dX: [batch_size, input_features]
// @returns layer.dW: [output_features, input_features]
Data1D linear_backward(LinearLayer* layer, Data1D* dY) {
    assert(layer->dW != NULL && "This linear layer wasn't initialized with the with_gradient flag");
    assert(dY->n == layer->out && "The gradient from the next layer has not the same number of features than this layer");
    assert(dY->b == layer->X.b && "The gradient from the next layer is not computed for the same number of batch than the input of this layer");

    Data1D dX = CreateData1D(layer->X.n, layer->X.b);
    init_matrix(dX.mat, 0.0, dX.b, dX.n);
    init_matrix(layer->dW, 0.0, layer->out, layer->in);
    // dX = dY * layer.w
    matrix_mul_2d(dY->mat, layer->w, dX.mat, dX.b, layer->out, dX.n);
    // dW = dY.T * X
    matrix_mul_2d_T1(dY->mat, layer->X.mat, layer->dW, layer->out, dY->b, layer->in);
    return dX;
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

// Creates data tensor of shape [batch_size, features]
Data1D CreateData1D(int features, int batch_size) {
    Data1D d;
    d.n = features;
    d.b = batch_size;
    d.mat = fmatrix_allocate_2d(batch_size, features);
    return d;
}

void DestroyData1D(Data1D* d) {
    if (d->mat == NULL) {
        return;
    }

    free_fmatrix_2d(d->mat);
    d->mat = NULL;
    printf("destroyed: %p\n", d->mat);
}

// Creates data tensor of shape [batch_size, channels, size, size]
Data2D CreateData2D(int size, int batch_size, int channels) {
    Data2D d;
    d.size = size;
    d.b = batch_size;
    d.c = channels;
    d.data = square_allocate_2d(batch_size, channels);
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < channels; j++)
            d.data[i][j] = CreateSquareMatrix(size);
    return d;
}

// Creates data tensor of shape [batch_size, channels, size, size]
Data2D CreateData2DZeros(int size, int batch_size, int channels) {
    Data2D d;
    d.size = size;
    d.b = batch_size;
    d.c = channels;
    d.data = square_allocate_2d(batch_size, channels);
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < channels; j++)
            d.data[i][j] = CreateZerosMatrix(size);
    return d;
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
}

// @param d: Data2D of shape [batch, channels, size, size]
// @returns d_: Data1D of shape [batch, channels*size*size]
Data1D flatten(Data2D d2) {
    Data1D d1 = CreateData1D(d2.c*d2.size*d2.size, d2.b);

    #pragma omp parallel for
    for (int k = 0; k < d2.b; k++) {
        Square* db2 = d2.data[k];
        float* db1 = d1.mat[k];
        for (int l = 0; l < d2.c; l++){
            float** curr = db2[l].mat;
            int c_shift = l * d2.size * d2.size;
            for (int i = 0; i < d2.size; i++) {
                int vpos = i * d2.size;
                for (int j = 0; j < d2.size; j++)
                    db1[c_shift + vpos + j] = curr[i][j];
            }
        }
    }

    return d1;
}

// @param d: Data1D of shape [batch, size]
// @param channels: number of channels to create in the output tensor
// @returns d_: Data2D of shape [batch, channels, sqrt(size/channels), sqrt(size/channels)]
Data2D unflatten(Data1D d1, int channels) {
    int size = (int)sqrt((double) (d1.n/channels));
    assert(pow((double) size, 2)*channels == (double) d1.n && "Can't unflatten data with non square number of features with this number of channels");

    Data2D d2 = CreateData2D(size, d1.b, channels);

    #pragma omp parallel for
    for (int k = 0; k < d2.b; k++) {
        Square* db2 = d2.data[k];
        float* db1 = d1.mat[k];
        for (int l = 0; l < d2.c; l++){
            float** curr = db2[l].mat;
            int c_shift = l * d2.size * d2.size;
            for (int i = 0; i < d2.size; i++) {
                int vpos = i * d2.size;
                for (int j = 0; j < d2.size; j++)
                    curr[i][j] = db1[c_shift + vpos + j];
            }
        }
    }

    return d2;
}

LinearLayer CreateLinearLayer(int in_channels, int out_channels, int with_gradient) {
    LinearLayer l;
    l.w = fmatrix_allocate_2d(out_channels, in_channels);
    l.in = in_channels;
    l.out = out_channels;

    l.dW = with_gradient ? fmatrix_allocate_2d(out_channels, in_channels) : NULL;
    l.X.mat = NULL;

    return l;
}

void RandomInitLinearLayer(LinearLayer* l) {
    random_init_matrix(l->w, l->out, l->in);
}

void DestroyLinearLayer(LinearLayer* layer) {
    free_fmatrix_2d(layer->w);
    if (layer->dW != NULL) free_fmatrix_2d(layer->dW);
    layer->w = NULL;
    layer->dW = NULL;
}

ConvLayer CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient) {
    ConvLayer c;
    c.w = square_allocate_2d(out_channels, in_channels);
    c.dW = with_gradient ? square_allocate_2d(out_channels, in_channels) : NULL;
    c.X.data = NULL;

    for (int i; i<in_channels; i++) 
        for (int j; j<out_channels; j++) {
            c.w[i][j] = CreateSquareMatrix(size);
            if (with_gradient) c.dW[i][j] = CreateSquareMatrix(size);
        }

    c.in = in_channels;
    c.out = out_channels;
    c.size = size;
}

void RandomInitConvLayer(ConvLayer* c) {
    for (int i; i<c->out; i++)
        for (int j; j<c->in; j++)
            random_init_matrix(c->w[i][j].mat, c->size, c->size);
}

void DestroyConvLayer(ConvLayer* c) {
    for (int i; i<c->in; i++) 
        for (int j; j<c->out; j++) {
            DestroySquareMatrix(c->w[i][j]);
            if (c->dW != NULL) {
                DestroySquareMatrix(c->dW[i][j]);
            } 
        }
    free(c->w[0]);
    free(c->w);
    c->w = NULL;
    if (c->dW != NULL) {
        free(c->dW[0]);
        free(c->dW);
        c->dW = NULL;
    }
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


// Returns Y size of a convulation of a W of size filter_size
// on an X of size X_size
int get_output_size(int input_size, int filter_size) {
    return input_size - filter_size + 1;
} 