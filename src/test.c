#include "utils.h"
#include "serialize.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./test /path/to/model");
        exit(1);
    }

    Network* net = read_newtork(argv[1], FALSE);
    load_data();
    test_epoch(net, 0);
    printf("\n");
    return 0;
}