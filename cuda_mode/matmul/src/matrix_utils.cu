#include "matrix_utils.cuh"

template <typename T>
void randomize_matrix(T *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for(int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (float)(rand() % 5);
        if(std::is_same<T, half>::value) {
            tmp = __float2half(tmp);
        }
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs) {
    int i;
    fs << std::setprecision(2) << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for(i = 0; i < M * N; i++) {
        if((i + 1) % N == 0)
            fs << std::setw(5) << (float)A[i]; // Set field width and write the value
        else
            fs << std::setw(5) << (float)A[i] << ", ";
        if((i + 1) % N == 0) {
            if(i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}

template <typename T>
bool verify_matrix(T *matRef, T *matOut, int N) {
    double diff = 0.0;
    int i;
    for(i = 0; i < N; i++) {
        diff = std::fabs((float)matRef[i] - (float)matOut[i]);
        if(diff > 0.01) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n", (float)matRef[i], (float)matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

template void randomize_matrix<float>(float *mat, int N);
template void randomize_matrix<half>(half *mat, int N);
template void print_matrix<float>(const float *A, int M, int N, std::ofstream &fs);
template void print_matrix<half>(const half *A, int M, int N, std::ofstream &fs);
template bool verify_matrix<float>(float *matRef, float *matOut, int N);
template bool verify_matrix<half>(half *matRef, half *matOut, int N);