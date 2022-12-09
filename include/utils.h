#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

#include "data.h"
#include "layers.h"

#define NBCHAR 200
#define NUM_TRAIN 60000

void load_data();

long long timeInMilliseconds(void);
void test_epoch(Network* net, int epoch);
void train_epoch(Network* net, float lr, int* indices, int epoch, int batch_size);
void shuffle(int *arr, size_t n);
void load_batch(Data2D* inputs, int* gt, int pos, int batch, float data[][784], int labels[], int* indices);

Data2D* LoadImage(char* name);
void SoftmaxTransform(Data1D* data);


float** LoadImagePgm(char* name,int *length,int *width);
void SaveImagePgm(char* name,float** mat,int length,int width);
