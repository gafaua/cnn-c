#include "network.h"

Network* CreateNetwork() {
    Network* net = (Network*) malloc(sizeof(Network));
    net->first = NULL;
    net->last = NULL;
    return net;
}

void AddToNetwork(Network* network, LayerNode* node) {
    if (network->first == NULL) {
        network->first = node;
        network->last = node;
    }
    else {
        node->previous = network->last;
        network->last->next = node;
        network->last = node;
    }
}

DataType* network_forward(LayerNode* node, DataType* input) {
    DataType* output;
    switch (node->type)
    {
    case Linear:
        assert(*input == D1D && "The input of a Linear Layer must be a Data1D");
        output = (DataType*) linear_forward((LinearLayer*) node, (Data1D*) input);
        break;
    case Conv:
        assert(*input == D2D && "The input of a Conv Layer must be a Data2D");
        output = (DataType*) conv_forward((ConvLayer*) node, (Data2D*) input);
        break;
    case Flatten:
        assert(*input == D2D && "The input of a Flatten Layer must be a Data2D");
        output = (DataType*) flatten((Data2D*) input);
        break;
    case Unflatten:
        assert(*input == D1D && "The input of an Unflatten Layer must be a Data1D");
        assert(node->next != NULL && node->next->type == Conv &&"An Unflatten Layer must lead to a Conv Layer");
        int channels = ((ConvLayer*) node->next)->in;
        output = (DataType*) unflatten((Data1D*) input, channels);
        break;
    default:
        break;
    }
    
    if (node->next != NULL) {
        return network_forward(node->next, output);
    }

    return output;
}

DataType* network_backward(LayerNode* node, DataType* dY) {
    DataType* dX;
    switch (node->type)
    {
    case Linear:
        assert(*dY == D1D && "The output of a Linear Layer must be a Data1D");
        dX = (DataType*) linear_backward((LinearLayer*) node, (Data1D*) dY);
        break;
    case Conv:
        assert(*dY == D2D && "The output of a Conv Layer must be a Data2D");
        dX = (DataType*) conv_backward((ConvLayer*) node, (Data2D*) dY);
        break;
    case Flatten:
        assert(*dY == D1D && "The output of a Flatten Layer must be a Data1D");
        assert(node->previous != NULL && node->previous->type == Conv &&"A flatten Layer must follow a Conv Layer");
        int channels = ((ConvLayer*) node->previous)->out;
        dX = (DataType*) unflatten((Data1D*) dY, channels);
        break;
    case Unflatten:
        assert(*dY == D2D && "The output of an Unflatten Layer must be a Data2D");
        dX = (DataType*) flatten((Data2D*) dY);
        break;
    default:
        break;
    }

    if (node->previous != NULL) {
        return network_backward(node->previous, dX);
    }

    return dX;
}

void DestroyNetwork(Network* network) {
    LayerNode* node = network->first;
    while (node != NULL) {
        LayerNode* next = node->next;

        switch (node->type)
        {
        case Linear:
            DestroyLinearLayer((LinearLayer*) node);
            break;
        case Conv:
            DestroyConvLayer((ConvLayer*) node);
            break;
        default:
            break;
        }

        node = next;
    }
    free(network);
}
