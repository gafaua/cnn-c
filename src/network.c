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

DataType* network_forward(Network* network, DataType* input) {
    LayerNode* node = network->first;
    DataType* X;
    if (*input == D1D) X = (DataType*) CopyData1D((Data1D*) input);
    else if (*input == D2D) X = (DataType*) CopyData2D((Data2D*) input);
    DataType* output;

    while(node != NULL) {
        switch (node->type)
        {
        case Linear:
            assert(*X == D1D && "The input of a Linear Layer must be a Data1D");
            output = (DataType*) linear_forward((LinearLayer*) node, (Data1D*) X);
            break;
        case Conv:
            assert(*X == D2D && "The input of a Conv Layer must be a Data2D");
            output = (DataType*) conv_forward((ConvLayer*) node, (Data2D*) X);
            break;
        case Flatten:
            assert(*X == D2D && "The input of a Flatten Layer must be a Data2D");
            output = (DataType*) flatten((Data2D*) X);
            DestroyData2D((Data2D*) X); // Destroy input since not used for backprop
            break;
        case Unflatten:
            assert(*X == D1D && "The input of an Unflatten Layer must be a Data1D");
            assert(node->next != NULL && node->next->type == Conv &&"An Unflatten Layer must lead to a Conv Layer");
            int channels = ((ConvLayer*) node->next)->in;
            output = (DataType*) unflatten((Data1D*) X, channels);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        default:
            break;
        }

        node = node->next;
        X = output;
    }

    return output;
}


void network_backward(Network* network, DataType* dY) {
    LayerNode* node = network->last;
    DataType* dX;

    while (node != NULL) {
        switch (node->type) {
        case Linear:
            assert(*dY == D1D && "The output of a Linear Layer must be a Data1D");
            dX = (DataType*) linear_backward((LinearLayer*) node, (Data1D*) dY);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Conv:
            assert(*dY == D2D && "The output of a Conv Layer must be a Data2D");
            dX = (DataType*) conv_backward((ConvLayer*) node, (Data2D*) dY);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        case Flatten:
            assert(*dY == D1D && "The output of a Flatten Layer must be a Data1D");
            assert(node->previous != NULL && node->previous->type == Conv &&"A flatten Layer must follow a Conv Layer");
            int channels = ((ConvLayer*) node->previous)->out;
            dX = (DataType*) unflatten((Data1D*) dY, channels);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Unflatten:
            assert(*dY == D2D && "The output of an Unflatten Layer must be a Data2D");
            dX = (DataType*) flatten((Data2D*) dY);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        default:
            break;
        }

        node = node->previous;
        dY = dX;
    }

    // Destroy last gradient, not useful
    if (*dX == D1D) DestroyData1D((Data1D*) dY);
    else if (*dX == D2D) DestroyData2D((Data2D*) dY);
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
        case Flatten:
        case Unflatten:
            DestroyLayerNode(node);
            break;
        default:
            break;
        }

        node = next;
    }
    free(network);
}

Network* CreateNetworkMNIST(int with_gradients) {
    Network* net = CreateNetwork();
    // Inputs: Data2D of size [b, 1, 28, 28]
    AddToNetwork(net, (LayerNode*) CreateConvLayer(1, 5, 5, with_gradients, TRUE));
    // 24
    AddToNetwork(net, (LayerNode*) CreateConvLayer(5, 10, 5, with_gradients, TRUE));
    // 20
    AddToNetwork(net, (LayerNode*) CreateConvLayer(10, 32, 5, with_gradients, TRUE));
    // 16
    AddToNetwork(net, (LayerNode*) CreateConvLayer(32, 10, 5, with_gradients, TRUE));
    // 12
    AddToNetwork(net, CreateFlattenLayer());
    // 10 * 12 * 12
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(10 * 12 * 12, 100, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 10, with_gradients, TRUE));
    return net;
}
