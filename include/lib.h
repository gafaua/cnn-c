#ifndef LIBCNN_H
#define LIBCNN_H

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
    Data1D X;  // Last input passed through this layer
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
    Data2D X;   // Last input passed through this layer

    float (*activation)(float);
} ConvLayer;

Network* CreateNetwork();
void AddToNetwork(Network* network, LayerNode* node);
DataType* network_forward(LayerNode* node, DataType* data);
DataType* network_backward(LayerNode* node, DataType* data);
void DestroyNetwork(Network* network);

Data2D* conv_forward(ConvLayer* layer, Data2D* input);
Data2D* conv_backward(ConvLayer* layer, Data2D* dY);

Data1D* linear_forward(LinearLayer* layer, Data1D* input);
Data1D* linear_backward(LinearLayer* layer, Data1D* dY);

float ReLU(float val);
float ReLU_backward(float val);
float Identity(float val);

float CrossEntropyForward(Data1D* y_hat, int* y);

Data1D* CreateData1D(int features, int batch_size);
void DestroyData1D(Data1D* d);
Data2D* CreateData2D(int size, int batch_size, int channels);
void RandomInitData2D(Data2D* d);
Data2D* CreateData2DZeros(int size, int batch_size, int channels);
void ClearData2D(Data2D* d);
void DestroyData2D(Data2D* d);

Data1D* flatten(Data2D* d);
Data2D* unflatten(Data1D* d, int channels);

LayerNode* CreateFlattenLayer();
LayerNode* CreateUnflattenLayer();
LinearLayer* CreateLinearLayer(int in, int out, int with_gradient, int random);
void RandomInitLinearLayer(LinearLayer* l);
void DestroyLinearLayer(LinearLayer* layer);
ConvLayer* CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient, int random);
void RandomInitConvLayer(ConvLayer* c);
void DestroyConvLayer(ConvLayer* c);

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
void print_data1d(Data1D* d);
void print_data2d(Data2D* d);
void print_conv_layer(ConvLayer* layer);
void print_square(Square s);
int get_output_size(int input_size, int filter_size);


#endif