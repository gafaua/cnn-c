#ifndef LIBCNN_H
#define LIBCNN_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct SquareMatrix {
    float** mat;
    int shape;
} Square;

typedef struct BackwardMatrices {
    Square dX;
    Square dW;
} BackwardPassResult;

typedef struct LinearLayer {
    float** w; // Matrix of size in * out
    int in;
    int out;
    float (*activation)(float);
} LinearLayer;

typedef struct ConvLayer {
    int in;
    int out;
    int shape;
    Square* kernels; // in * out kernels of size shape*shape

    float (*activation)(float);
} ConvLayer;

void conv_forward(Square input, Square filter, Square output, float (*activation)(float));
BackwardPassResult conv_backward(Square dY, Square X, Square W);

float ReLU(float val);
float ReLU_backward(float val);
float Identity(float val);

Square CreateSquareMatrix(int size);
Square CreateZerosMatrix(int size);
Square CopySquareMatrix(Square sq);
void DestroySquareMatrix(Square s);
void init_square(Square sq, float val);

float** fmatrix_allocate_2d(int vsize,int hsize);
void free_fmatrix_2d(float** pmat);
void init_matrix(float** m, float val, int h, int w);
void print_matrix(float** m, int h, int w);
void print_square(Square s);
int get_output_shape(int input_size, int filter_size);


#endif