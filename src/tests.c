#include "tests.h"

void test_functions_memory() {
    int n = 10;
    int b = 2;

    int size2d = 5;
    int c = 2;

    Data1D* d1d = CreateData1D(n, b);
    random_init_matrix(d1d->mat, b, n);
    Data2D* d2d = CreateData2D(size2d, b, c);
    RandomInitData2D(d2d);
    printf("Creating Linear...");
    LinearLayer* ll = CreateLinearLayer(n, 5, TRUE, TRUE);
    printf("Ok\nRandom Linear Init...");
    printf("Ok\nCreating Conv...");
    ConvLayer* cl = CreateConvLayer(c, 1, 3, TRUE, TRUE);
    printf("Ok\nTesting LinearLayer forward...");
    Data1D* d1d_y = linear_forward(ll, d1d);
    printf("Ok\nTesting Linear Backward...");
    d1d_y->mat[0][0] = -1;
    Data1D* d1d_y_ = linear_backward(ll, d1d_y);
    printf("Ok\nTesting print Data1d...");
    print_data1d(d1d_y_);
    printf("Ok\nTesting Conv Forward...");
    Data2D* d2d_y = conv_forward(cl, d2d);
    printf("Ok\nTesting Conv Backward...");
    Data2D* d2d_y_ = conv_backward(cl, d2d_y);
    print_data2d(d2d_y_);
    printf("Ok\nTesting Flatten...");
    Data1D* d2d_flat = flatten(d2d_y_);
    printf("Ok\nTesting Unflatten...");
    Data2D* d1d_unflat = unflatten(d2d_flat, c);

    printf("Ok\nDestroying Data1D...");
    DestroyData1D(d1d);
    DestroyData1D(d1d_y);
    DestroyData1D(d1d_y_);
    DestroyData1D(d2d_flat);
    printf("Ok\nDestroying Data2D...");
    DestroyData2D(d2d);
    DestroyData2D(d2d_y);
    DestroyData2D(d2d_y_);
    DestroyData2D(d1d_unflat);
    printf("Ok\nDestroying Linear Layer...");
    DestroyLinearLayer(ll);
    printf("Ok\nDestroying Conv Layer...");
    DestroyConvLayer(cl);
    printf("\nOk\n");

    printf("Individual functions are memory safe\n");
}

void test_network() {
    int in = 10;
    int b = 1;

    printf("Testing Network Creation...");
    Network* net = CreateNetwork();
    printf("Ok\nTesting Add to network...");
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(in, 100, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 500, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(500, 100, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 32, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(32, 200, TRUE, TRUE));
    AddToNetwork(net, CreateUnflattenLayer());
    AddToNetwork(net, (LayerNode*) CreateConvLayer(2, 5, 3, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateConvLayer(5, 2, 3, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateConvLayer(2, 5, 3, TRUE, TRUE));
    AddToNetwork(net, CreateFlattenLayer());
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(80, 10, TRUE, TRUE));
    printf("Ok\nTesting Forward pass...");

    Data1D* inputs = CreateData1D(in, b);
    random_init_matrix(inputs->mat, b, in);

    Data1D* outputs = (Data1D*) network_forward(net->first, (DataType*) inputs);
    printf("Ok\nTesting Backward pass...");
    network_backward(net->last, (DataType*) outputs);
    printf("Ok\nTesting Destroy Network...");

    DestroyNetwork(net);
    printf("Ok\nNetwork passed all tests.\n");
}

void test_all() {
    test_functions_memory();
    test_network();
}