#include <omp.h>

namespace Transpose {
    namespace Regular {

        template<typename T>
        void transpose(T* dest, const T *src, int rows, int cols) {
            int prod = rows*cols;

            for (int m = 0; m < prod; m++) {
                int i = m / rows;
                int j = m % rows;
                dest[m] = src[i + cols * j];
            }
        }

        template<typename T>
        void transpose_parallel(T* dest, const T *src, int rows, int cols) {
            int prod = rows*cols;

#pragma omp parallel for
            for (int m = 0; m < prod; m++) {
                int i = m / rows;
                int j = m % rows;
                dest[m] = src[i + cols * j];
            }
        }

    }
}