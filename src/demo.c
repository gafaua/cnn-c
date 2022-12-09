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

    long long tmp = timeInMilliseconds();
    Data1D* output = (Data1D*) network_forward(net, (DataType*) image);
    SaveImagePgm("oui", image->data[0][0].mat, 28, 28);
    SoftmaxTransform(output);
    print_data1d(output);
    int n = argmax_vector(output->mat[0], output->n);
    printf("Network prediction: %d at %.2f%% confidence, prediction computed in %lldms\n", n, output->mat[0][n] * 100, timeInMilliseconds()-tmp);

    DestroyNetwork(net);
    DestroyData1D(output);
    DestroyData2D(image);

    return 0;
}
