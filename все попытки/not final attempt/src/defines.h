#pragma once

#define ALLIGNMENT 32 // avx2, не avx512

// Убрать потом:
#define BIGCACHE_SZ 16000000/6 // Размер L3-Cache моего компа (разделённый на число потоков)
#define THREAD_NUM 12

#ifndef SMALLCACHE_SZ // Размер L1-Cache
    #define SMALLCACHE_SZ 32000 
#endif


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

