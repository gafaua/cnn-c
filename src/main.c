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
#include "serialize.h"

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

void train_epoch(Network* net, float lr, int* indices, int epoch, int batch_size) {
    int num_batch = NUM_TRAIN / batch_size;
    
    Data2D* inputs = CreateData2D(28, batch_size, 1);
    int* gt = (int*) malloc(sizeof(int) * batch_size);

    Data1D* outputs;
    LossResult loss;
    setbuf(stdout, NULL);

    float loss_sum = 0.0;
    float acc_sum = 0.0;
    float time_elapsed = 0;
    int cnt = 0;
    long long eta;
    int min, s;
    long long start = timeInMilliseconds();

    for (int i = 0; i < num_batch; i++) {
        printf("\r[%d] Train: Loading -> ", epoch);
        load_batch(inputs, gt, i, batch_size, train_image, train_label, indices);

        printf("Forward -> ");
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);

        loss = CrossEntropy(net, outputs, gt);

        printf("Backward -> ");
        network_backward(net, (DataType*) loss.dL, lr);

        DestroyData1D(outputs);
        if (loss.value != INFINITY) {
            loss_sum += loss.value;
            acc_sum += loss.accuracy;
            cnt++;
        }

        time_elapsed = (timeInMilliseconds() - start) / 1000;
        eta = ((num_batch - i - 1) * (time_elapsed/(i+1)));
        printf(" [%d/%d] | [ETA %2d min %2d s > %2d min %2d s] | loss: %f (%f) | acc: %f (%f)", i, num_batch, (int)eta/60, (int)eta%60, (int)time_elapsed/60, (int)time_elapsed%60, loss.value, loss_sum / cnt, loss.accuracy, acc_sum / cnt);
    }
}


void test_epoch(Network* net, int epoch) {
    int batch = 100;
    int num_batch = NUM_TEST / batch;
    
    Data2D* inputs = CreateData2D(28, batch, 1);
    int* gt = (int*) malloc(sizeof(int) * batch);

    Data1D* outputs;
    LossResult loss;
    setbuf(stdout, NULL);

    float loss_sum = 0.0;
    float acc_sum = 0.0;
    float time_elapsed = 0;
    int cnt = 0;
    long long eta;
    int min, s;
    long long start = timeInMilliseconds();

    int* indices = (int*) malloc(sizeof(int)*NUM_TEST);
    for (int i = 0; i < NUM_TEST; i++) indices[i] = i;

    for (int i = 0; i < num_batch; i++) {
        printf("\r[%d] Test: Loading -> ", epoch);
        load_batch(inputs, gt, i, batch, test_image, test_label, indices);

        printf("Forward -> ");
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);

        loss = CrossEntropy(net, outputs, gt);

        DestroyData1D(outputs);

        if (loss.value != INFINITY) {
            loss_sum += loss.value;
            acc_sum += loss.accuracy;
            cnt++;
        }

        time_elapsed = (timeInMilliseconds() - start) / 1000;
        eta = ((num_batch - i - 1) * (time_elapsed/(i+1)));
        printf(" [%d/%d] | [ETA %2d min %2d s > %2d min %2d s] | loss: %f (%f) | acc: %f (%f)", i, num_batch, (int)eta/60, (int)eta%60, (int)time_elapsed/60, (int)time_elapsed%60, loss.value, loss_sum / cnt, loss.accuracy, acc_sum / cnt);
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

    int* indices = (int*) malloc(sizeof(int)*NUM_TRAIN);
    for (int i = 0; i < NUM_TRAIN; i++) indices[i] = i;

    float lr = 1e-3;
    int num_epoch = 10;

    load_mnist();
    char base_name[] = "checkfc";
    char name[20];
    Network* net = CreateNetworkMNIST(TRUE);
    //Network* net = read_newtork(name, TRUE);
    for (int i = 1; i <= num_epoch; i++)
    {
        shuffle(indices, NUM_TRAIN);
        train_epoch(net, lr, indices, i, 64);
        printf("\n");

        snprintf(name, 20, "%s_%d.bin", base_name, i);
        save_newtork(net, name); 

        test_epoch(net, i);
        if (i%3 == 0) {
            lr *= 0.1;
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
