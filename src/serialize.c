#include "serialize.h"


ConvLayer* read_conv_layer(FILE* file, int with_gradient) {
    // Just needs to save dimensions and weights
    int out, in, size;
    // Dimensions
    fread(&out, sizeof(int), 1, file);
    fread(&in, sizeof(int), 1, file);
    fread(&size, sizeof(int), 1, file);
    printf("Conv [%d x %d x %d]\n", in, out, size);

    ConvLayer* layer = CreateConvLayer(in, out, size, with_gradient, FALSE);
    // Weights
    for (int i = 0; i < layer->out; i++)
        for (int j = 0; j < layer->in; j++)
            for (int k = 0; k < layer->size; k++) {
                fread(layer->w[i][j].mat[k], sizeof(float), size, file);
            }

    // Bias
    // fread(layer->b, sizeof(float), layer->out, file);

    return layer;
}

void write_conv_layer(FILE* file, ConvLayer* layer) {
    // Just needs to save dimensions and weights
    // Dimensions
    fwrite(&layer->out, sizeof(int), 1, file);
    fwrite(&layer->in, sizeof(int), 1, file);
    fwrite(&layer->size, sizeof(int), 1, file);
    // Weights
    for (int i = 0; i < layer->out; i++)
        for (int j = 0; j < layer->in; j++)
            for (int k = 0; k < layer->size; k++) {
                fwrite(layer->w[i][j].mat[k], sizeof(float), layer->size, file);
            }
    // Bias
    // fwrite(layer->b, sizeof(float), layer->out, file);
}

void write_linear_layer(FILE* file, LinearLayer* layer) {
    // Just needs to save dimensions and weights
    // Dimensions
    fwrite(&layer->out, sizeof(int), 1, file);
    fwrite(&layer->in, sizeof(int), 1, file);
    // Weights
    for (int i = 0; i < layer->out; i++)
        fwrite(layer->w[i], sizeof(float), layer->in, file);
    // Bias
    fwrite(layer->b, sizeof(float), layer->out, file);
}

LinearLayer* read_linear_layer(FILE* file, int with_gradient) {
    // Just needs to save dimensions and weights
    int out, in;
    // Dimensions
    fread(&out, sizeof(int), 1, file);
    fread(&in, sizeof(int), 1, file);

    printf("Lin [%d x %d]\n", in, out);

    LinearLayer* layer = CreateLinearLayer(in, out, with_gradient, FALSE);
    // Weights
    for (int i = 0; i < layer->out; i++)
        fread(layer->w[i], sizeof(float), layer->in, file);
    // Bias
    fread(layer->b, sizeof(float), layer->out, file);

    return layer;
}

void save_newtork(Network* net, char* filename) {
    LayerNode* node = net->first;
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error saving network to file.\n");
        exit(1);
    }

    printf("Saving network to file: %s\n", filename);

    // Write number of nodes in the network
    fwrite(&net->size, sizeof(int), 1, file);

    while(node != NULL) {
        fwrite(&node->type, sizeof(NodeType), 1, file);
        switch(node->type) {
            case Linear:
                write_linear_layer(file, (LinearLayer*) node);
                break;
            case Conv:
                write_conv_layer(file, (ConvLayer*) node);
                break;
            case Flatten:
            case Unflatten:
                fwrite(&((ViewLayer*) node)->channels, sizeof(int), 1, file);
                break;
            case MaxPool:
                fwrite(&((MaxPoolLayer*) node)->size, sizeof(int), 1, file);
                break;
            default:
                break;
        }

        node = node->next;
    }
    fclose(file);
}


Network* read_newtork(char* filename, int with_gradient) {
    FILE* file = fopen(filename, "rb");

    if (file == NULL) {
        printf("Error reading network from file.\n");
        exit(1);
    }

    int size;
    fread(&size, sizeof(int), 1, file);
    assert(size > 0);

    printf("Reading network %s with %d layers:\n", filename, size);

    Network* net = CreateNetwork();
    NodeType n;
    int buff;

    while(size > 0) {
        fread(&n, sizeof(NodeType), 1, file);

        switch(n) {
            case Linear:
                AddToNetwork(net, (LayerNode*) read_linear_layer(file, with_gradient));
                break;
            case Conv:
                AddToNetwork(net, (LayerNode*) read_conv_layer(file, with_gradient));
                break;
            case Flatten:
                fread(&buff, sizeof(int), 1, file);
                printf("Flat [%d]\n", buff);
                AddToNetwork(net, (LayerNode*) CreateFlattenLayer(buff));
                break;
            case Unflatten:
                fread(&buff, sizeof(int), 1, file);
                printf("Uflat [%d]\n", buff);
                AddToNetwork(net, (LayerNode*) CreateUnflattenLayer(buff));
                break;
            case MaxPool:
                fread(&buff, sizeof(int), 1, file);
                printf("MaxPool [%d]\n", buff);
                AddToNetwork(net, (LayerNode*) CreateMaxPoolLayer(buff, with_gradient));
                break;
            case ReLU1D:
                printf("Relu1\n");
                AddToNetwork(net, (LayerNode*) CreateReLU1DLayer(with_gradient));
                break;
            case ReLU2D:
                printf("Relu2\n");
                AddToNetwork(net, (LayerNode*) CreateReLU2DLayer(with_gradient));
                break;
            case Tanh1D:
                printf("Tanh1\n");
                AddToNetwork(net, (LayerNode*) CreateTanh1DLayer(with_gradient));
                break;
            case Tanh2D:
                printf("Tanh2\n");
                AddToNetwork(net, (LayerNode*) CreateTanh2DLayer(with_gradient));
                break;

            default:
                break;
        }

        size--;
    }
    fclose(file);

    return net;
}