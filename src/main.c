#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#include "lib.h"
#include "tests.h"
#define TRUE 1
#define FALSE 0

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);

    long long start, end; 
    start = timeInMilliseconds();
    // start timer. 
    test_all();

    end = timeInMilliseconds();
    // print_data1d(inputs);
    // print_data1d(outputs);

    long long time_taken = (end - start);

    printf("Time taken for processing: %lld ms.\n", time_taken);

    printf("Done.\n");
}