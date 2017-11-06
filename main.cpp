// Based on: https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c

#include <cstdlib>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <bitset>
#include <stdint.h>
#include <errno.h>
#include <chrono>

#include "regular.h"
#include "sse_float.h"
#include "sse_byte.h"

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

using namespace std;


void print_data(uint8_t* data, size_t rows, size_t cols) {
    for(size_t i=0; i<rows*cols; i++) {
        cout << (int)data[i] << " ";
        if(!((i+1) % rows)) cout << endl;
    }
}

void print_separator(size_t size, bool add_endl = true) {
    for(size_t i=0; i<size; i++) cout << "*";
    cout << endl;
}

int main(int argc, char** argv) {
    //int rows = 128, cols = 3*224*224; // Resnet
    int rows = 16, cols = 16; // Test
    int size = rows*cols;
    auto *data_in = new uint8_t[size];
    auto *data_out = new uint8_t[size];

    for(int i=1; i<=size; i++) data_in[i-1] = i;

    // Regular
    auto start_time = chrono::high_resolution_clock::now();
    Transpose::Regular::transpose<uint8_t>(data_out, data_in, rows, cols);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Regular: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;

    // Regular + parallel
    start_time = chrono::high_resolution_clock::now();
    Transpose::Regular::transpose_parallel<uint8_t>(data_out, data_in, rows, cols);
    end_time = chrono::high_resolution_clock::now();
    cout << "Regular parallel: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;

/*    
    // Intrinsics + SSE + float
    auto *dataInF = new TEST_TYPE[rows*cols];
    auto *dataOutF = new TEST_TYPE[rows*cols];
    int lda = ROUND_UP(cols, 16);
    int ldb = ROUND_UP(rows, 16);
    int block_size = 64;
    for(int i=1; i<=output_size; i++) dataInF[i-1] = i;
    start_time = chrono::high_resolution_clock::now();
    Transpose::SSE::transpose((float*)dataInF, (float*)dataOutF, rows, cols, lda, ldb, block_size);
    end_time = chrono::high_resolution_clock::now();
    cout << "Intrinsics: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    delete dataInF;
    delete dataOutF;
*/

    // SSE for byte
    start_time = chrono::high_resolution_clock::now();
    Transpose::SSE::transpose(data_out, data_in, rows, cols);
    end_time = chrono::high_resolution_clock::now();
    cout << "SSE: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    print_data(data_in, rows, cols);
    print_separator(cols);
    print_data(data_out, rows, cols);

/*
    // AVX2
//    dataInF = new float[rows*cols];
//    dataOutF = new float[rows*cols];
//    lda = ROUND_UP(cols, 16);
//    ldb = ROUND_UP(rows, 16);
//    block_size = 64;
//    for(int i=1; i<=output_size; i++) dataInF[i-1] = i;
//    start_time = chrono::high_resolution_clock::now();
//    cout << rows << " " << cols << " " << endl;
//    transpose_block_AVX2(dataInF, dataOutF, rows, cols, lda, ldb, block_size);
//    end_time = chrono::high_resolution_clock::now();
//    cout << "Intrinsics: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
//    delete dataInF;
//    delete dataOutF;
*/

    delete[] data_in;
    delete[] data_out;

    return 0;
}
