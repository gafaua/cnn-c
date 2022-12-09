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
#include "serialize.h"
#include "utils.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./demo <path/to/model.bin> <path/to/image.png>");
        exit(1);
    }

    Network* net = read_newtork(argv[1], FALSE);
    Data2D* image = LoadImage(argv[2]);
    // for (int i = 0; i< 28; i++) {
    //     for (int j = 0; j< 28; j++) {
    //         printf("%.2f ", image->data[0][0].mat[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // load_mnist();
    // int t = 700;
    // for (int i = 0; i< 28; i++) {
    //     for (int j = 0; j< 28; j++) {
    //         image->data[0][0].mat[i][j] = test_image[t][i*28 + j];
    //     }
    // }

    // printf("Ground Truth: %d\n", test_label[t]);
    Data1D* output = (Data1D*) network_forward(net, (DataType*) image);
    SaveImagePgm("oui", image->data[0][0].mat, 28, 28);
    SoftmaxTransform(output);
    print_data1d(output);
    printf("Network prediction: %d\n", argmax_vector(output->mat[0], output->n));

    DestroyNetwork(net);
    DestroyData1D(output);
    DestroyData2D(image);

    return 0;
}
