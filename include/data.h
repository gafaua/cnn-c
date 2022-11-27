#pragma once

#include "definitions.h"

Square CreateSquareMatrix(int size);
Square CreateZerosMatrix(int size);
Square CopySquareMatrix(Square sq);
void DestroySquareMatrix(Square s);
void init_square(Square sq, float val);

float** fmatrix_allocate_2d(int vsize,int hsize);
Square** square_allocate_2d(int vsize,int hsize);
Square* square_allocate_1d(int size);
void free_fmatrix_2d(float** pmat);
void init_matrix(float** m, float val, int h, int w);
void random_init_matrix(float** m, int h, int w);
void matrix_mul_2d(float** M1, float** M2, float** R, int a, int b, int c);
void matrix_mul_2d_T1(float** M1T, float** M2, float** R, int a, int b, int c);
void matrix_mul_2d_T2(float** M1, float** M2T, float** R, int a, int b, int c);

void print_matrix(float** m, int h, int w);

Data1D* CreateData1D(int features, int batch_size);
void DestroyData1D(Data1D* d);
Data2D* CreateData2D(int size, int batch_size, int channels);
void RandomInitData2D(Data2D* d);
Data2D* CreateData2DZeros(int size, int batch_size, int channels);
void ClearData2D(Data2D* d);
void DestroyData2D(Data2D* d);

void print_data1d(Data1D* d);
void print_data2d(Data2D* d);
void print_square(Square s);
