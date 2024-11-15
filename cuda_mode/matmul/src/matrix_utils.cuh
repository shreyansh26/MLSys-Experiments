#pragma once
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sys/time.h>

void randomize_matrix(float *mat, int N);

void print_matrix(const float *A, int M, int N, std::ofstream &fs);

bool verify_matrix(float *mat1, float *mat2, int N);