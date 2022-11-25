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

    // start timer. 
    int in = 10;
    int b = 1;

    LinearLayer* ll1 = CreateLinearLayer(in, 100, TRUE, TRUE);
    LinearLayer* ll2 = CreateLinearLayer(100, 500, TRUE, TRUE);
    LinearLayer* ll3 = CreateLinearLayer(500, 100, TRUE, TRUE);
    LinearLayer* ll4 = CreateLinearLayer(100, 32, TRUE, TRUE);
    LinearLayer* ll5 = CreateLinearLayer(32, 200, TRUE, TRUE);
    LayerNode* u1 = (LayerNode*) malloc(sizeof(LayerNode));
    u1->type = Unflatten;
    ConvLayer* cl1 = CreateConvLayer(2, 5, 3, TRUE, TRUE);
    ConvLayer* cl2 = CreateConvLayer(5, 2, 3, TRUE, TRUE);
    ConvLayer* cl3 = CreateConvLayer(2, 5, 3, TRUE, TRUE);
    LayerNode* f1 = (LayerNode*) malloc(sizeof(LayerNode));
    f1->type = Flatten;
    LinearLayer* ll6 = CreateLinearLayer(80, 10, TRUE, TRUE);

    ll1->node.next = (LayerNode*) ll2;
    ll2->node.next = (LayerNode*) ll3;
    ll3->node.next = (LayerNode*) ll4;
    ll4->node.next = (LayerNode*) ll5;
    ll5->node.next = u1;
    u1->next = (LayerNode*) cl1;
    cl1->node.next = (LayerNode*) cl2;
    cl2->node.next = (LayerNode*) cl3;
    cl3->node.next = f1;
    f1->next = (LayerNode*) ll6;

    Data1D* inputs = CreateData1D(in, b);
    random_init_matrix(inputs->mat, b, in);
    start = timeInMilliseconds(); 
    Data1D* outputs = (Data1D*) network_forward((LayerNode*) ll1, (DataType*) inputs);


    end = timeInMilliseconds();
    print_data2d(&cl2->X);
    // print_data1d(inputs);
    // print_data1d(outputs);

    long long time_taken = (end - start);

    printf("Time taken for processing: %lld ms.\n", time_taken);

    printf("Done.\n");
}