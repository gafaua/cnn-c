#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TRUE 1
#define FALSE 0

typedef enum {
    None,
    Linear,
    Conv,
    MaxPool,
    ReLU1D,
    ReLU2D,
    Flatten,
    Unflatten,
} NodeType;

typedef enum {
    D1D=1,
    D2D=2,
} DataType;

typedef struct SquareMatrix {
    float** mat;
    int size;
} Square;

typedef struct Data1D {
    DataType type;
    float** mat; // Matrix of size [b, n]
    int n;
    int b;
} Data1D;

typedef struct Data2D {
    DataType type;
    Square** data; // Equivalent of a tensor of size [b, c, size, size]
    int size;
    int c;
    int b;
} Data2D;

typedef struct LayerNode {
    struct LayerNode* previous;
    struct LayerNode* next;
    NodeType type;

} LayerNode;

typedef struct Network {
    LayerNode* first;
    LayerNode* last;
} Network;

typedef struct LinearLayer {
    LayerNode node;
    float** w;  // Matrix of size [out, in] * [in, b] -> [out, b]
    float** dW; // Gradient matrix of size [out, in]
    Data1D* X;  // Last input passed through this layer
    int in;
    int out;

    float (*activation)(float);
} LinearLayer;

typedef struct ConvLayer {
    LayerNode node;
    int in;
    int out;
    int size;
    Square** w;  // [out, in] kernels of size size*size
    Square** dW; // Gradient matrix of [out, in] kernels of size size*size
    Data2D* X;   // Last input passed through this layer

    float (*activation)(float);
} ConvLayer;

typedef struct MaxPoolLayer {
    LayerNode node;
    int size;
    Data2D* X;

} MaxPoolLayer;

typedef struct ReLU1DLayer {
    LayerNode node;
    Data1D* X;
    int with_gradient;
} ReLU1DLayer;

typedef struct ReLU2DLayer {
    LayerNode node;
    Data2D* X;
    int with_gradient;
} ReLU2DLayer;
