#include "matrix_utils.cuh"

void randomize_matrix(float *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for(int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
    int i;
    fs << std::setprecision(2) << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for(i = 0; i < M * N; i++) {
        if((i + 1) % N == 0)
            fs << std::setw(5) << A[i]; // Set field width and write the value
        else
            fs << std::setw(5) << A[i] << ", ";
        if((i + 1) % N == 0) {
            if(i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for(i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if(diff > 0.01) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n", matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}