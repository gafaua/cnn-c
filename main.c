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

    int b = 1;
    int n = 10;
    Data1D d = CreateData1D(n, b);
    init_matrix(d.mat, 1.0, b, n);
    LinearLayer l1 = CreateLinearLayer(n, 4, TRUE);
    RandomInitLinearLayer(l1);

    printf("Init done, starting processing\n");
    // start timer. 
    time(&start); 

    Data1D d2 = linear_forward(&l1, &d);
    Data2D d3 = unflatten(d2, 1);
    Data1D d4 = flatten(d3);
    printf("%d\n", l1.X.b);

    time(&end); 

    time_t time_taken;

    time_taken = (end - start);

    printf("Time taken for processing: %ld sec.\n", time_taken);
    print_matrix(d2.mat, d2.b, d2.n);
    print_matrix(d4.mat, d4.b, d4.n);
    print_matrix(d.mat, b, n);
    print_matrix(l1.X.mat, b, n);

    printf("Clearing memory...\n");

    DestroyData1D(&d);
    DestroyLinearLayer(&l1);
    DestroyData1D(&d2);

    printf("Done.\n");
}