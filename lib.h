#ifndef LIBCNN_H
#define LIBCNN_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct SquareMatrix {
    float** mat;
    int size;
} Square;

typedef struct BackwardMatrices {
    Square dX;
    Square dW;
} BackwardConvResult;

typedef struct Data1D {
    float** mat; // Matrix of size [b, n]
    int n;
    int b;
} Data1D;

typedef struct Data2D {
    Square** data; // Equivalent of a tensor of size [b, c, size, size]
    int size;
    int c;
    int b;
} Data2D;

typedef struct LinearLayer {
    float** w;  // Matrix of size [out, in] * [in, b] -> [out, b]
    float** dW; // Gradient matrix of size [out, in]
    int in;
    int out;
    float (*activation)(float);
} LinearLayer;

typedef struct ConvLayer {
    int in;
    int out;
    int size;
    Square** w;  // [out, in] kernels of size size*size
    Square** dW; // Gradient matrix of [out, in] kernels of size size*size

    float (*activation)(float);
} ConvLayer;

void convolution(Square X, Square W, Square Y);
Data2D conv_forward(ConvLayer layer, Data2D inputs);
BackwardConvResult conv_backward(Square dY, Square X, Square W);

Data1D linear_forward(LinearLayer layer, Data1D inputs);
Data1D linear_backward(Data1D dY, Data1D X, LinearLayer layer);

float ReLU(float val);
float ReLU_backward(float val);
float Identity(float val);

Data1D CreateData1D(int features, int batch_size);
void DestroyData1D(Data1D d);
Data2D CreateData2D(int size, int batch_size, int channels);
Data2D CreateData2DZeros(int size, int batch_size, int channels);
void DestroyData2D(Data2D d);

Data1D flatten(Data2D d);
Data2D unflatten(Data1D d, int channels);

LinearLayer CreateLinearLayer(int in_channels, int out_channels, int with_gradient);
void DestroyLinearLayer(LinearLayer layer);
ConvLayer CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient);
void DestroyConvLayer(ConvLayer c);

Square CreateSquareMatrix(int size);
Square CreateZerosMatrix(int size);
Square CopySquareMatrix(Square sq);
void DestroySquareMatrix(Square s);
void init_square(Square sq, float val);

float** fmatrix_allocate_2d(int vsize,int hsize);
Square** square_allocate_2d(int vsize,int hsize);
void free_fmatrix_2d(float** pmat);
void init_matrix(float** m, float val, int h, int w);
void random_init_matrix(float** m, int h, int w);
void matrix_mul_2d(float** M1, float** M2, float** R, int a, int b, int c);
void matrix_mul_2d_T1(float** M1T, float** M2, float** R, int a, int b, int c);
void matrix_mul_2d_T2(float** M1, float** M2T, float** R, int a, int b, int c);

void print_matrix(float** m, int h, int w);
void print_square(Square s);
int get_output_size(int input_size, int filter_size);


#endif