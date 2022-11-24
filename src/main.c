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
    int n = 5;

    printf("Init done, starting processing\n");
    // start timer. 
    time(&start); 

    int gt[2] = {4, 2};
    Data1D d = CreateData1D(n, b);
    random_init_matrix(d.mat, b, n);

    float loss = CrossEntropyForward(&d, (int *)&gt);
    print_data1d(&d);
    printf("Loss: %f\n", loss);

    time(&end);

    time_t time_taken;

    time_taken = (end - start);

    printf("Time taken for processing: %ld sec.\n", time_taken);

    printf("Clearing memory...\n");

    DestroyData1D(&d);

    printf("Done.\n");
}