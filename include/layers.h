#pragma once

#include "definitions.h"
#include "data.h"

Data2D* conv_forward(ConvLayer* layer, Data2D* input);
Data2D* conv_backward(ConvLayer* layer, Data2D* dY);

Data1D* linear_forward(LinearLayer* layer, Data1D* input);
Data1D* linear_backward(LinearLayer* layer, Data1D* dY);

Data2D* max_pool_forward(MaxPoolLayer* layer, Data2D* input);
Data2D* max_pool_backward(MaxPoolLayer* layer, Data2D* dY);

float ReLU(float val);
float ReLU_backward(float val);
float Identity(float val);

float CrossEntropyForward(Data1D* y_hat, int* y);

Data1D* flatten(Data2D* d);
Data2D* unflatten(Data1D* d, int channels);

LayerNode* CreateFlattenLayer();
LayerNode* CreateUnflattenLayer();
void DestroyLayerNode(LayerNode* node);

LinearLayer* CreateLinearLayer(int in, int out, int with_gradient, int random);
void RandomInitLinearLayer(LinearLayer* l);
void LearnLinearLayer(LinearLayer* l, float learning_rate);
void DestroyLinearLayer(LinearLayer* layer);

ConvLayer* CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient, int random);
void RandomInitConvLayer(ConvLayer* c);
void LearnConvLayer(ConvLayer* c, float learning_rate);
void DestroyConvLayer(ConvLayer* c);

void print_conv_layer(ConvLayer* layer);
int get_output_size(int input_size, int filter_size);
