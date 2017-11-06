#include <stdint.h>
#include <immintrin.h>

namespace Transpose {
    namespace AVX2 {

        static inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh) {
            __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
            __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
            *vl = _mm256_unpacklo_epi32(va, vb);
            *vh = _mm256_unpackhi_epi32(va, vb);
        }

        static inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh) {
            __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
            __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
            *vl = _mm256_unpacklo_epi64(va, vb);
            *vh = _mm256_unpackhi_epi64(va, vb);
        }

        static inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh) {
            *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
            *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
        }

        // In place transpose of 8 x 8 int array

        static void Transpose_8_8(
                __m256i *v0,
                __m256i *v1,
                __m256i *v2,
                __m256i *v3,
                __m256i *v4,
                __m256i *v5,
                __m256i *v6,
                __m256i *v7) {
            __m256i w0, w1, w2, w3, w4, w5, w6, w7;
            __m256i x0, x1, x2, x3, x4, x5, x6, x7;

            _mm256_merge_epi32(*v0, *v1, &w0, &w1);
            _mm256_merge_epi32(*v2, *v3, &w2, &w3);
            _mm256_merge_epi32(*v4, *v5, &w4, &w5);
            _mm256_merge_epi32(*v6, *v7, &w6, &w7);

            _mm256_merge_epi64(w0, w2, &x0, &x1);
            _mm256_merge_epi64(w1, w3, &x2, &x3);
            _mm256_merge_epi64(w4, w6, &x4, &x5);
            _mm256_merge_epi64(w5, w7, &x6, &x7);

            _mm256_merge_si128(x0, x4, v0, v1);
            _mm256_merge_si128(x1, x5, v2, v3);
            _mm256_merge_si128(x2, x6, v4, v5);
            _mm256_merge_si128(x3, x7, v6, v7);
        }

        inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
            __t0 = _mm256_unpacklo_ps(row0, row1);
            __t1 = _mm256_unpackhi_ps(row0, row1);
            __t2 = _mm256_unpacklo_ps(row2, row3);
            __t3 = _mm256_unpackhi_ps(row2, row3);
            __t4 = _mm256_unpacklo_ps(row4, row5);
            __t5 = _mm256_unpackhi_ps(row4, row5);
            __t6 = _mm256_unpacklo_ps(row6, row7);
            __t7 = _mm256_unpackhi_ps(row6, row7);
            __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
            __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
            __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
            __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
            __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
            __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
            __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
            __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
            row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
            row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
            row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
            row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
            row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
            row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
            row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
            row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
        }

        inline void transpose8x8_avx(float *A, float *B, const int lda, const int ldb) {
            __m256 row0 = _mm256_load_ps(&A[0 * lda]);
            __m256 row1 = _mm256_load_ps(&A[1 * lda]);
            __m256 row2 = _mm256_load_ps(&A[2 * lda]);
            __m256 row3 = _mm256_load_ps(&A[3 * lda]);
            __m256 row4 = _mm256_load_ps(&A[4 * lda]);
            __m256 row5 = _mm256_load_ps(&A[5 * lda]);
            __m256 row6 = _mm256_load_ps(&A[6 * lda]);
            __m256 row7 = _mm256_load_ps(&A[7 * lda]);
            transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
            _mm256_store_ps(&B[0 * ldb], row0);
            _mm256_store_ps(&B[1 * ldb], row1);
            _mm256_store_ps(&B[2 * ldb], row2);
            _mm256_store_ps(&B[3 * ldb], row3);
            _mm256_store_ps(&B[4 * ldb], row4);
            _mm256_store_ps(&B[5 * ldb], row5);
            _mm256_store_ps(&B[6 * ldb], row6);
            _mm256_store_ps(&B[7 * ldb], row7);
        }

        void transpose_block_AVX2(float *A, float *B, const int n, const int m, const int lda, const int ldb, const int block_size) {
            //#pragma omp parallel for
            for (int i = 0; i < n; i += block_size) {
                for (int j = 0; j < m; j += block_size) {
                    int max_i2 = i + block_size < n ? i + block_size : n;
                    int max_j2 = j + block_size < m ? j + block_size : m;
                    for (int i2 = i; i2 < max_i2; i2 += 4) {
                        for (int j2 = j; j2 < max_j2; j2 += 4) {
                            transpose8x8_avx(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
                        }
                    }
                }
            }
        }

    }
}
