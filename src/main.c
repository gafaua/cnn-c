#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "data.h"
#include "layers.h"
#include "network.h"
#include "tests.h"
#include "serialize.h"
#include "utils.h"

#define TRUE 1
#define FALSE 0

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);
    // int seed = time(NULL);
    // printf("Setting random seed: %d\n", seed);
    // srand(seed);
    long long start, end; 
    start = timeInMilliseconds();

    int* indices = (int*) malloc(sizeof(int)*NUM_TRAIN);
    for (int i = 0; i < NUM_TRAIN; i++) indices[i] = i;

    float lr = 1e-3;
    int num_epoch = 10;

    load_data();
    char base_name[] = "model";
    char name[20];
    Network* net = CreateNetworkMNIST(TRUE);
    //Network* net = read_newtork("check_CNN_5.bin", TRUE);
    for (int i = 1; i <= num_epoch; i++)
    {
        shuffle(indices, NUM_TRAIN);
        train_epoch(net, lr, indices, i, 64);
        printf("\n");

        snprintf(name, 20, "%s_%d.bin", base_name, i);
        save_newtork(net, name); 

        test_epoch(net, i);
        if (i%2 == 0) {
            lr /= 5;
            printf("\nNew Learning Rate: %f", lr);
        }
        printf("\n");
    }

    // save image of first data in test dataset as .pgm file
    //save_mnist_pgm(train_image, test);

    // show all pixels and labels in test dataset
    // print_mnist_pixel(test_image, NUM_TEST);
    //print_mnist_label(test_label, NUM_TRAIN);

    end = timeInMilliseconds();
    // print_data1d(inputs);
    // print_data1d(outputs);
    free(indices);
    long long time_taken = (end - start);

    printf("Time taken for processing: %lld ms.\n", time_taken);

    printf("Done.\n");
    return 0;
}
