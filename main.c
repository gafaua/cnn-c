#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#include "lib.h"
#define TRUE 1
#define FALSE 0

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);

    time_t start, end; 

    int b = 2;
    int n = 10;
    Data1D d = CreateData1D(n, b);
    init_matrix(d.mat, 1.0, b, n);
    d.mat[0][0] = 0.0;
    LinearLayer l1 = CreateLinearLayer(n, 18, TRUE);

    RandomInitLinearLayer(&l1);

    printf("Init done, starting processing\n");
    // start timer. 
    time(&start); 

    Data1D d2 = linear_forward(&l1, &d);
    print_data1d(&d2);

    Data2D d3 = unflatten(d2, 2);
    print_data2d(&d3);

    Data1D d4 = flatten(d3);
    print_data1d(&d4);

    time(&end);

    time_t time_taken;

    time_taken = (end - start);

    printf("Time taken for processing: %ld sec.\n", time_taken);

    printf("Clearing memory...\n");

    DestroyData1D(&d);
    DestroyLinearLayer(&l1);
    DestroyData1D(&d2);

    printf("Done.\n");
}