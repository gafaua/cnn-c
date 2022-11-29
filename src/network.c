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
    DataType* output;

    while(node != NULL) {
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
            DestroyData2D((Data2D*) input); // Destroy input since not used for backprop
            break;
        case Unflatten:
            assert(*input == D1D && "The input of an Unflatten Layer must be a Data1D");
            assert(node->next != NULL && node->next->type == Conv &&"An Unflatten Layer must lead to a Conv Layer");
            int channels = ((ConvLayer*) node->next)->in;
            output = (DataType*) unflatten((Data1D*) input, channels);
            DestroyData1D((Data1D*) input); // Destroy input since not used for backprop
            break;
        default:
            break;
        }

        node = node->next;
        input = output;
    }

    return output;
}


void network_backward(Network* network, DataType* dY) {
    LayerNode* node = network->last;
    DataType* dX;

    while (node != NULL) {
        printf("%d\n", node->type);
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
            printf("Destroying Linear\n");
            DestroyLinearLayer((LinearLayer*) node);
            break;
        case Conv:
            printf("Destroying Conv\n");
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
