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
        printf("Usage: ./featuremaps <path/to/model.bin> <path/to/image.png>");
        exit(1);
    }

    Network* net = read_newtork(argv[1], TRUE);
    Data2D* image = LoadImage(argv[2]);

    long long tmp = timeInMilliseconds();
    Data1D* output = (Data1D*) network_forward(net, (DataType*) image);

    LayerNode* node = net->first;
    int num = 0;
    while(node != NULL) {
        switch (node->type) {
            case Conv:
                if (num == 0) { // skip first layer
                    num += 1;
                    break;
                }
                save_feature_maps(((ConvLayer*) node)->X, num);
                num += 1;
                break;
            case Flatten:
                int channels = ((ViewLayer*) node)->channels;
                node = node->next;
                save_feature_maps(unflatten(((LinearLayer*)node)->X, channels), num);
                num += 1;
            default:
                break;
        }
        node = node->next;
    }

    SoftmaxTransform(output);
    print_data1d(output);
    int n = argmax_vector(output->mat[0], output->n);
    printf("Network prediction: %d at %.2f%% confidence, prediction computed in %lldms\n", n, output->mat[0][n] * 100, timeInMilliseconds()-tmp);

    DestroyNetwork(net);
    DestroyData1D(output);
    DestroyData2D(image);

    return 0;
}
