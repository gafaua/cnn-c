#pragma once

#include "definitions.h"
#include "layers.h"
#include "data.h"

Network* CreateNetwork();
void AddToNetwork(Network* network, LayerNode* node);
DataType* network_forward(Network* network, DataType* input);
void network_backward(Network* network, DataType* dY);
void DestroyNetwork(Network* network);
