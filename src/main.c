#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#include "lib.h"
#include "tests.h"
#define TRUE 1
#define FALSE 0

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);

    long long start, end; 
    start = timeInMilliseconds();

    // start timer. 
    int in = 10;
    int b = 1;

    Network* net = CreateNetwork();
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

    Data1D* inputs = CreateData1D(in, b);
    random_init_matrix(inputs->mat, b, in);

    printf("Forward pass\n");
    Data1D* outputs = (Data1D*) network_forward(net->first, (DataType*) inputs);
    printf("Backward pass\n");
    network_backward(net->last, (DataType*) outputs);

    DestroyNetwork(net->first);

    end = timeInMilliseconds();
    // print_data1d(inputs);
    // print_data1d(outputs);

    long long time_taken = (end - start);

    printf("Time taken for processing: %lld ms.\n", time_taken);

    printf("Done.\n");
}