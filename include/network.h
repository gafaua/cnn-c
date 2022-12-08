#pragma once

#include "definitions.h"
#include "layers.h"
#include "data.h"

Network* CreateNetwork();
void AddToNetwork(Network* network, LayerNode* node);
DataType* network_forward(Network* network, DataType* input);
void network_backward(Network* network, DataType* dY, float lr);
void DestroyNetwork(Network* network);
Network* CreateNetworkMNIST(int with_gradients);
Network* CreateNetworkMNIST_FC(int with_gradients);
float GetLayersNorm(Network* net);
