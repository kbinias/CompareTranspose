#include <cstdlib>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <bitset>
#include <stdint.h>
#include <errno.h>
#include <chrono>
#include <omp.h>
#include <xmmintrin.h>

using namespace std;

typedef uint8_t  TEST_TYPE;

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

// Naive
template<typename T>
T *transpose_naive(const T *src, int rows, int cols) {
    int prod = rows*cols;
    T* dest = new T [prod];

    for(int m = 0; m<prod; m++) {
        int i = m/rows;
        int j = m%rows;
        dest[m] = src[i + cols*j];
    }

    return dest;
}

// Naive + parallel
template<typename T>
T *transpose_parallel(const T *src, int rows, int cols) {
    int prod = rows*cols;
    T* dest = new T [prod];

    #pragma omp parallel for
    for(int m = 0; m<prod; m++) {
        int i = m/rows;
        int j = m%rows;
        dest[m] = src[i + cols*j];
    }

    return dest;
}

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}

// Intrinsics + parallel
inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}

// Based on: https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
int main(int argc, char** argv) {
    int output_size = 10;
    int rows = 128, cols = 3*224*224; // Resnet
    auto *data = new TEST_TYPE[rows*cols];
    
    for(int i=1; i<=output_size; i++) data[i-1] = i;
    
    // Naive
    auto start_time = chrono::high_resolution_clock::now();
    auto trans = transpose_naive<TEST_TYPE>((TEST_TYPE*)data, rows, cols);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Naive: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    delete trans;

    // Naive + parallel
    start_time = chrono::high_resolution_clock::now();
    trans = transpose_parallel<TEST_TYPE>((TEST_TYPE*)data, rows, cols);
    end_time = chrono::high_resolution_clock::now();
    cout << "Parallel: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    delete trans;
    
    // Intrinsics + parallel
    auto *dataInF = new float[rows*cols];
    auto *dataOutF = new float[rows*cols];
    int lda = ROUND_UP(cols, 16);
    int ldb = ROUND_UP(rows, 16);
    int block_size = 64;
    for(int i=1; i<=output_size; i++) dataInF[i-1] = i;
    start_time = chrono::high_resolution_clock::now();
    transpose_block_SSE4x4(dataInF, dataOutF, rows, cols, lda, ldb, block_size);
    end_time = chrono::high_resolution_clock::now();
    cout << "Intrinsics: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    delete dataInF;
    delete dataOutF;

    delete data;

    return 0;
}
