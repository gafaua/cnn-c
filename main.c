#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#include "lib.h"

int main(int argc,char** argv) {
    int p = omp_get_max_threads();
    printf("Max number of threads used: %d\n\n", p);
    int a = 3000;
    int b = 2000;
    int c = 3000;
    float** M1 = fmatrix_allocate_2d(a, b);
    float** M2 = fmatrix_allocate_2d(b, c);
    float** R = fmatrix_allocate_2d(a, c);

    random_init_matrix(M1, a, b);
    random_init_matrix(M2, b, c);
    init_matrix(R, 0.0, a, c);


    // print_matrix(M1, a, b);
    // print_matrix(M2, c, b);

    time_t start, end; 
    printf("Init done, starting multiplication\n");
    // start timer. 
    time(&start); 
    
    matrix_mul_2d(M1, M2, R, a, b, c);

    time(&end); 
    
    time_t time_taken;

    time_taken = (end - start);

    printf("Time taken for [%d, %d] x [%d, %d]: %ld sec.\n", a, b, b, c, time_taken);
    // print_matrix(R, a, c);

    printf("Done.\n");
}