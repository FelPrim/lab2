#pragma once

#ifndef SMALLCACHE_SZ // Размер L1-Cache
#define SMALLCACHE_SZ 32000 
#endif

#define BIGCACHE_SZ 16000000 // Размер L3-Cache моего компа. Убрать потом

// BIGCACHE_PERCENTAGE = BIGCACHE_FRACTION / 256 * 100%
#ifndef BIGCACHE_FRACTION
    #define BIGCACHE_FRACTION 192
#endif

#ifndef BIGCACHE_SZ // Размер L3-Cache (если есть) и L2-Cache (если L3 нет)
#define BIGCACHE_SZ 2097152 // Размер L2-Cache моего ноута
#endif

#ifndef THREAD_NUM
#define THREAD_NUM 2 // Число потоков на моём ноуте. Вряд ли у кого-то есть комп хуже
#endif

#ifndef ALU_NUM
#define ALU_NUM 4 // Число ALU для одного потока. 
                  // По-хорошему, их нужно учитывать самому, 
                  // но я понадеюсь на оптимизации компилятора -
                  // Не просто ж так я компилирую с Ofast, нативными march, mtune и pgo
#endif

#include "magic_numbers.h"
// Для 32000 CACHEBLOCK_LENGTH равен 56

#ifndef CACHEBLOCK_LENGTH
    #error "CACHEBLOCK_LENGTH not defined"
#endif

typedef double aligned_double alignas(32); // 32=8*4, 8 - размер double, 4 - число double в одном __m256d для avx2

constexpr int CACHEBLOCK_SZ = CACHEBLOCK_LENGTH * CACHEBLOCK_LENGTH;

typedef aligned_double CacheBlock; // CacheBlock* всегда указывает на double[CACHEBLOCK_SZ]

// Ухудшенная версия std::vector из C++
struct DataVector{
    aligned_double *doubles;
    size_t size;
    size_t capacity;
}; // typedef для слабых

// Что-то типа append_range, но вместо записи значений просто выделяется (резервируется) память
// По-хорошему quantity должно быть кратно CACHEBLOCK_LENGTH
void data_reserve(struct DataVector* const data, const size_t quantity);

void data_construct(struct DataVector* const data, const size_t capacity);
void data_destruct(struct DataVector* const data);
void data_shrink(struct DataVector* const data);

int calculate_ROWHEIGHT(const size_t nonzero, const size_t length, const size_t height);

typedef struct SparseMatrix{
    aligned_double *data; // я мог бы засунуть DataVector 
    unsigned int *indices;
    unsigned int *indptr;
    size_t count; // count == data.size/CACHEBLOCK_SZ
} SparseMatrix;

