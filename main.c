#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "lib.h"

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);

    Square input  = CreateSquareMatrix(5);
    Square filter = CreateSquareMatrix(3);

    Square output = CreateSquareMatrix(get_output_shape(input.shape, filter.shape));
    
    init_square(input, 0.5);

    input.mat[1][1] = 2;
    input.mat[0][1] = 4;

    init_matrix(filter.mat, 1.0, filter.shape, filter.shape);
    filter.mat[0][1] = -1;
    filter.mat[1][0] = -1;
    filter.mat[1][1] = -1;

    print_square(input);
    print_square(filter);

    conv_forward(input, filter, output, &Identity);
    print_square(output);

    Square dY = CreateZerosMatrix(3);
    BackwardPassResult res = conv_backward(output, input, filter);

    print_square(res.dW);
    print_square(res.dX);

    DestroySquareMatrix(input);
    DestroySquareMatrix(output);
    DestroySquareMatrix(filter);

    printf("Done.\n");
}