#pragma once
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sys/time.h>
#include <cuda_fp16.h>

template <typename T>
void randomize_matrix(T *mat, int N);

template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs);

template <typename T>
bool verify_matrix(T *mat1, T *mat2, int N, bool diff_layout = false,double atol = 1e-3, double rtol = 1e-2);