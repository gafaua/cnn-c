#pragma once

#include "definitions.h"
#include "layers.h"
#include "data.h"

Network* CreateNetwork();
void AddToNetwork(Network* network, LayerNode* node);
DataType* network_forward(LayerNode* node, DataType* data);
DataType* network_backward(LayerNode* node, DataType* data);
void DestroyNetwork(Network* network);
