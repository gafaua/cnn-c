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
            int channels = ((ViewLayer*) node)->channels;
            output = (DataType*) unflatten((Data1D*) X, channels);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        case ReLU1D:
            assert(*X == D1D && "The input of a RELU1D Layer must be a Data1D");
            output = (DataType*) relu_1d_forward((Activation1DLayer*) node, (Data1D*) X);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        case ReLU2D:
            assert(*X == D2D && "The input of a RELU2D Layer must be a Data2D");
            output = (DataType*) relu_2d_forward((Activation2DLayer*) node, (Data2D*) X);
            DestroyData2D((Data2D*) X); // Destroy input since not used for backprop
            break;
        case Tanh1D:
            assert(*X == D1D && "The input of a Tanh1D Layer must be a Data1D");
            output = (DataType*) tanh_1d_forward((Activation1DLayer*) node, (Data1D*) X);
            DestroyData1D((Data1D*) X); // Destroy input since not used for backprop
            break;
        case Tanh2D:
            assert(*X == D2D && "The input of a Tanh2D Layer must be a Data2D");
            output = (DataType*) tanh_2d_forward((Activation2DLayer*) node, (Data2D*) X);
            DestroyData2D((Data2D*) X); // Destroy input since not used for backprop
            break;
        case MaxPool:
            assert(*X == D2D && "The input of a MaxPool Layer must be a Data2D");
            output = (DataType*) max_pool_forward((MaxPoolLayer*) node, (Data2D*) X);
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


void network_backward(Network* network, DataType* dY, float lr) {
    LayerNode* node = network->last;
    DataType* dX;

    while (node != NULL) {
        switch (node->type) {
        case Linear:
            assert(*dY == D1D && "The output of a Linear Layer must be a Data1D");
            dX = (DataType*) linear_backward((LinearLayer*) node, (Data1D*) dY, lr);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Conv:
            assert(*dY == D2D && "The output of a Conv Layer must be a Data2D");
            dX = (DataType*) conv_backward((ConvLayer*) node, (Data2D*) dY, lr);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        case Flatten:
            assert(*dY == D1D && "The output of a Flatten Layer must be a Data1D");
            assert(node->previous != NULL &&"A flatten Layer must follow a Conv Layer");
            int channels = ((ViewLayer*) node)->channels;
            dX = (DataType*) unflatten((Data1D*) dY, channels);
            DestroyData1D((Data1D*) dY); // Last dY not useful anymore
            break;
        case Unflatten:
            assert(*dY == D2D && "The output of an Unflatten Layer must be a Data2D");
            dX = (DataType*) flatten((Data2D*) dY);
            DestroyData2D((Data2D*) dY); // Last dY not useful anymore
            break;
        case ReLU1D:
            assert(*dY == D1D && "The output of a ReLU1D Layer must be a Data1D");
            dX = (DataType*) relu_1d_backward((Activation1DLayer*) node, (Data1D*) dY);
            break;
        case ReLU2D:
            assert(*dY == D2D && "The output of a ReLU2D Layer must be a Data2D");
            dX = (DataType*) relu_2d_backward((Activation2DLayer*) node, (Data2D*) dY);
            break;
        case Tanh1D:
            assert(*dY == D1D && "The output of a Tanh1D Layer must be a Data1D");
            dX = (DataType*) tanh_1d_backward((Activation1DLayer*) node, (Data1D*) dY);
            break;
        case Tanh2D:
            assert(*dY == D2D && "The output of a Tanh2D Layer must be a Data2D");
            dX = (DataType*) tanh_2d_backward((Activation2DLayer*) node, (Data2D*) dY);
            break;
        case MaxPool:
            assert(*dY == D2D && "The output of a MaxPool Layer must be a Data2D");
            dX = (DataType*) max_pool_backward((MaxPoolLayer*) node, (Data2D*) dY);
            DestroyData2D((Data2D*) dY);
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

        switch (node->type) {
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
        case Tanh1D:
            DestroyActivation1DLayer((Activation1DLayer*) node);
            break;
        case ReLU2D:
        case Tanh2D:
            DestroyActivation2DLayer((Activation2DLayer*) node);
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
    // // Inputs: Data2D of size [b, 1, 28, 28]
    // AddToNetwork(net, (LayerNode*) CreateConvLayer(1, 32, 3, with_gradients, TRUE));
    // AddToNetwork(net, (LayerNode*) CreateActivation2DLayer(with_gradients));
    // // 26 ->
    // AddToNetwork(net, (LayerNode*) CreateMaxPoolLayer(2, TRUE));
    // // 13 ->
    // AddToNetwork(net, (LayerNode*) CreateConvLayer(32, 64, 4, with_gradients, TRUE));
    // AddToNetwork(net, (LayerNode*) CreateActivation2DLayer(with_gradients));
    // // 10 ->
    // AddToNetwork(net, (LayerNode*) CreateMaxPoolLayer(2, TRUE));
    // // 5 ->
    // AddToNetwork(net, (LayerNode*) CreateFlattenLayer(64));
    // // 5 * 5 * 5
    // AddToNetwork(net, (LayerNode*) CreateLinearLayer(64 * 5* 5, 10, with_gradients, TRUE));

    // 28 ->
    AddToNetwork(net, (LayerNode*) CreateConvLayer(1, 16, 5, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateTanh2DLayer(with_gradients));
    AddToNetwork(net, (LayerNode*) CreateMaxPoolLayer(2, with_gradients));
    // 12 ->
    AddToNetwork(net, (LayerNode*) CreateConvLayer(16, 32, 5, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateTanh2DLayer(with_gradients));
    AddToNetwork(net, (LayerNode*) CreateMaxPoolLayer(2, with_gradients));
    // 4 ->
    AddToNetwork(net, (LayerNode*) CreateFlattenLayer(32));
    // 4 * 4 * 32 ->
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(4*4*32, 256, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateTanh1DLayer(with_gradients));

    AddToNetwork(net, (LayerNode*) CreateLinearLayer(256, 128, with_gradients, TRUE));
    AddToNetwork(net, (LayerNode*) CreateTanh1DLayer(with_gradients));

    AddToNetwork(net, (LayerNode*) CreateLinearLayer(128, 10, with_gradients, TRUE));

    return net;
}
