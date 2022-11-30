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
            printf("L");
            assert(*X == D1D && "The input of a Linear Layer must be a Data1D");
            output = (DataType*) linear_forward((LinearLayer*) node, (Data1D*) X);
            break;
        case Conv:
            printf("C");
            assert(*X == D2D && "The input of a Conv Layer must be a Data2D");
            output = (DataType*) conv_forward((ConvLayer*) node, (Data2D*) X);
            break;
        case Flatten:
            printf("F");
            assert(*X == D2D && "The input of a Flatten Layer must be a Data2D");
            output = (DataType*) flatten((Data2D*) X);
            DestroyData2D((Data2D*) X); // Destroy input since not used for backprop
            break;
        case Unflatten:
            printf("U");
            assert(*X == D1D && "The input of an Unflatten Layer must be a Data1D");
            assert(node->next != NULL && node->next->type == Conv &&"An Unflatten Layer must lead to a Conv Layer");
            int channels = ((ViewLayer*) node)->channels;
            output = (DataType*) unflatten((Data1D*) X, channels);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        case ReLU1D:
            printf("R1");
            assert(*X == D1D && "The input of a RELU1D Layer must be a Data1D");
            output = (DataType*) relu_1d_forward((ReLU1DLayer*) node, (Data1D*) X);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        case ReLU2D:
            printf("R2");
            assert(*X == D2D && "The input of a RELU2D Layer must be a Data2D");
            output = (DataType*) relu_2d_forward((ReLU2DLayer*) node, (Data2D*) X);
            DestroyData2D((Data2D*) X); // Destroy input since not used for backprop
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
            printf("L");
            assert(*dY == D1D && "The output of a Linear Layer must be a Data1D");
            dX = (DataType*) linear_backward((LinearLayer*) node, (Data1D*) dY);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Conv:
            printf("C");
            assert(*dY == D2D && "The output of a Conv Layer must be a Data2D");
            dX = (DataType*) conv_backward((ConvLayer*) node, (Data2D*) dY);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        case Flatten:
            printf("F");
            assert(*dY == D1D && "The output of a Flatten Layer must be a Data1D");
            assert(node->previous != NULL &&"A flatten Layer must follow a Conv Layer");
            int channels = ((ViewLayer*) node)->channels;
            dX = (DataType*) unflatten((Data1D*) dY, channels);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Unflatten:
            printf("U");
            assert(*dY == D2D && "The output of an Unflatten Layer must be a Data2D");
            dX = (DataType*) flatten((Data2D*) dY);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        case ReLU1D:
            printf("R1");
            assert(*dY == D1D && "The output of a ReLU1D Layer must be a Data1D");
            dX = (DataType*) relu_1d_backward((ReLU1DLayer*) node, (Data1D*) dY);
            break;
        case ReLU2D:
            printf("R2");
            assert(*dY == D2D && "The output of a ReLU2D Layer must be a Data2D");
            dX = (DataType*) relu_2d_backward((ReLU2DLayer*) node, (Data2D*) dY);
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
            DestroyViewLayer((ViewLayer*) node);
            break;
        case ReLU1D:
            DestroyReLU1DLayer((ReLU1DLayer*) node);
            break;
        case ReLU2D:
            DestroyReLU2DLayer((ReLU2DLayer*) node);
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
    AddToNetwork(net, (LayerNode*) CreateReLU2DLayer(with_gradients));
    // 24
    AddToNetwork(net, (LayerNode*) CreateConvLayer(5, 10, 5, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateReLU2DLayer(with_gradients));
    // 20
    AddToNetwork(net, (LayerNode*) CreateConvLayer(10, 32, 5, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateReLU2DLayer(with_gradients));
    // 16
    AddToNetwork(net, (LayerNode*) CreateConvLayer(32, 10, 5, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateReLU2DLayer(with_gradients));
    // 12
    AddToNetwork(net, (LayerNode*) CreateFlattenLayer(10));
    // 10 * 12 * 12
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(10 * 12 * 12, 100, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateReLU1DLayer(with_gradients));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 10, with_gradients, TRUE));
    return net;
}
