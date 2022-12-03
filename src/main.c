#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#include "data.h"
#include "layers.h"
#include "network.h"
#include "tests.h"
#include "mnist.h"

#define TRUE 1
#define FALSE 0

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

void load_dataset() {
    load_mnist();

}

/// @brief Populates inputs and labels with data for train/test
/// @param inputs 
/// @param labels 
/// @param i 
/// @param batch 
void load_batch(Data2D* inputs, int* gt, int pos, int batch, float data[][SIZE], int labels[], int* indices) {
    int start = pos*batch;

    #pragma omp parallel for 
    for (int i = 0; i < batch; i++) {
        int idx = indices[i+start];
        gt[i] = labels[idx];
        for (int j = 0; j < 28; j++)
            for (int k = 0; k < 28; k++)
                inputs->data[i][0].mat[j][k] = data[idx][k + j*28];
    }

}

void shuffle(int *arr, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        //srand(time(NULL));
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}

void train_epoch(Network* net, float lr, int* indices) {
    int batch = 32;
    int num_batch = NUM_TRAIN / batch;
    
    long long checkpoint; 

    Data2D* inputs = CreateData2D(28, batch, 1);
    int* gt = (int*) malloc(sizeof(int) * batch);

    Data1D* outputs;
    LossResult loss;
    setbuf(stdout, NULL);

    checkpoint = timeInMilliseconds();
    float loss_sum = 0.0;
    int cnt = 0;
    long long eta;
    for (int i = 0; i < num_batch; i++) {
        printf("\rLoading -> ");
        load_batch(inputs, gt, i, batch, train_image, train_label, indices);

        printf("Forward -> ");
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);

        loss = CrossEntropy(outputs, gt);

        printf("Backward -> ");
        network_backward(net, (DataType*) loss.dL, lr);

        DestroyData1D(outputs);
        if (loss.value != INFINITY) {
            loss_sum += loss.value;
            cnt++;
        }
        eta = ((num_batch - i - 1) * (timeInMilliseconds() - checkpoint )) / 1000;
        printf(" [%d/%d] | eta: %llds | loss: %f (%f)", i, num_batch, eta, loss.value, loss_sum / cnt);
        checkpoint = timeInMilliseconds();
    }
}


int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);
    // int seed = time(NULL);
    // printf("Setting random seed: %d\n", seed);
    // srand(seed);
    long long start, end; 
    start = timeInMilliseconds();

    load_mnist();

    int* indices = (int*) malloc(sizeof(int)*NUM_TRAIN);
    for (int i = 0; i < NUM_TRAIN; i++) indices[i] = i;

    Network* net = CreateNetworkMNIST(TRUE);
    float lr = 1e-7;
    int num_epoch = 5;

    for (int i = 0; i < num_epoch; i++)
    {
        shuffle(indices, NUM_TRAIN);

        train_epoch(net, lr, indices);
        lr *= 0.8;
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
}
