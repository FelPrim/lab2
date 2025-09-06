#pragma once

#include "useful_stuff.h"
#include "defines.h"

typedef double aligned_double alignas(ALLIGNMENT);

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
HF aligned_double* data_reserve(struct DataVector* const data, const size_t quantity);

HF void data_construct(struct DataVector* const data, const size_t capacity);

HF void data_destruct(struct DataVector* const data);
HF void data_shrink(struct DataVector* const data);

// HF int calculate_ROWHEIGHT(const size_t nonzero, const size_t length, const size_t height);

typedef struct SparseMatrix{
    aligned_double *data; // я мог бы засунуть DataVector 
    alignas(ALLIGNMENT) unsigned int *indices;
    alignas(ALLIGNMENT) unsigned int *indptr;
    size_t count; // count == data.size/CACHEBLOCK_SZ
} SparseMatrix;

