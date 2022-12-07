#pragma once

#include "definitions.h"
#include "data.h"
#include "network.h"

Data2D* conv_forward(ConvLayer* layer, Data2D* input);
Data2D* conv_backward(ConvLayer* layer, Data2D* dY, float lr);

Data1D* linear_forward(LinearLayer* layer, Data1D* input);
Data1D* linear_backward(LinearLayer* layer, Data1D* dY, float lr);

Data2D* max_pool_forward(MaxPoolLayer* layer, Data2D* input);
Data2D* max_pool_backward(MaxPoolLayer* layer, Data2D* dY);

Data2D* tanh_2d_forward(Activation2DLayer* layer, Data2D* input);
Data2D* tanh_2d_backward(Activation2DLayer* layer, Data2D* dY);

Data1D* tanh_1d_forward(Activation1DLayer* layer, Data1D* input);
Data1D* tanh_1d_backward(Activation1DLayer* layer, Data1D* dY);

Data2D* relu_2d_forward(Activation2DLayer* layer, Data2D* input);
Data2D* relu_2d_backward(Activation2DLayer* layer, Data2D* dY);

Data1D* relu_1d_forward(Activation1DLayer* layer, Data1D* input);
Data1D* relu_1d_backward(Activation1DLayer* layer, Data1D* dY);

float ReLU(float val);
float ReLU_backward(float val);
float Identity(float val);

LossResult CrossEntropy(Network* net, Data1D* y_hat, int* y) ;

Data1D* flatten(Data2D* d);
Data2D* unflatten(Data1D* d, int channels);

ViewLayer* CreateFlattenLayer(int channels);
ViewLayer* CreateUnflattenLayer(int channels);
void DestroyViewLayer(ViewLayer* node);

MaxPoolLayer* CreateMaxPoolLayer(int size, int with_gradient);
void DestroyMaxPoolLayer(MaxPoolLayer* layer);

Activation1DLayer* CreateReLU1DLayer(int with_gradient);
Activation1DLayer* CreateTanh1DLayer(int with_gradient);
void DestroyActivation1DLayer(Activation1DLayer* layer);

Activation2DLayer* CreateReLU2DLayer(int with_gradient);
Activation2DLayer* CreateTanh2DLayer(int with_gradient);
void DestroyActivation2DLayer(Activation2DLayer* layer);

LinearLayer* CreateLinearLayer(int in, int out, int with_gradient, int random);
void RandomInitLinearLayer(LinearLayer* l);
void LearnLinearLayer(LinearLayer* l, float learning_rate);
float GetLinearLayerNorm(LinearLayer* l);
void DestroyLinearLayer(LinearLayer* layer);

ConvLayer* CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient, int random);
void RandomInitConvLayer(ConvLayer* c);
void LearnConvLayer(ConvLayer* c, float learning_rate);
float GetConvLayerNorm(ConvLayer* l);
void DestroyConvLayer(ConvLayer* c);

void print_conv_layer(ConvLayer* layer);
int get_output_size(int input_size, int filter_size);
int get_input_size(int output_size, int filter_size);
