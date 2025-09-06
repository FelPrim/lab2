#pragma once

// SMALLCACHE_PERCENTAGE = SMALLCACHE_FRACTION / 256 * 100%
#ifndef SMALLCACHE_FRACTION
#define SMALLCACHE_FRACTION 210
#endif

// К сожалению, в C23 нельзя навесить constexpr на функцию... 
// CACHEBLOCK_LENGTH - это всего лишь наибольшее кратное 4 число,
// непревышающее sqrt(SMALLCACHE_FRACTION / 256 * SMALLCACHE_SZ / 8) - 1
#if SMALLCACHE_PERCENTAGE == 210
    #if SMALLCACHE_SZ < 244
        #error "Too small Cache size"
    #elif SMALLCACHE_SZ < 790
        #define CACHEBLOCK_LENGTH 4
    #elif SMALLCACHE_SZ < 1649
        #define CACHEBLOCK_LENGTH 8
    #elif SMALLCACHE_SZ < 2819
        #define CACHEBLOCK_LENGTH 12
    #elif SMALLCACHE_SZ < 4301
        #define CACHEBLOCK_LENGTH 16
    #elif SMALLCACHE_SZ < 6096
        #define CACHEBLOCK_LENGTH 20
    #elif SMALLCACHE_SZ < 8202
        #define CACHEBLOCK_LENGTH 24
    #elif SMALLCACHE_SZ < 10621
        #define CACHEBLOCK_LENGTH 28
    #elif SMALLCACHE_SZ < 13352
        #define CACHEBLOCK_LENGTH 32
    #elif SMALLCACHE_SZ < 16394
        #define CACHEBLOCK_LENGTH 36
    #elif SMALLCACHE_SZ < 19749
        #define CACHEBLOCK_LENGTH 40
    #elif SMALLCACHE_SZ < 23416
        #define CACHEBLOCK_LENGTH 44
    #elif SMALLCACHE_SZ < 27395
        #define CACHEBLOCK_LENGTH 48
    #elif SMALLCACHE_SZ < 31686
        #define CACHEBLOCK_LENGTH 52
    #elif SMALLCACHE_SZ < 36289
        #define CACHEBLOCK_LENGTH 56
    #elif SMALLCACHE_SZ < 41204
        #define CACHEBLOCK_LENGTH 60
    #elif SMALLCACHE_SZ < 46432
        #define CACHEBLOCK_LENGTH 64
    #elif SMALLCACHE_SZ < 51971
        #define CACHEBLOCK_LENGTH 68
    #elif SMALLCACHE_SZ < 57822
        #define CACHEBLOCK_LENGTH 72
    #elif SMALLCACHE_SZ < 63986
        #define CACHEBLOCK_LENGTH 76
    #elif SMALLCACHE_SZ < 70461
        #define CACHEBLOCK_LENGTH 80
    #elif SMALLCACHE_SZ < 77249
        #define CACHEBLOCK_LENGTH 84
    #elif SMALLCACHE_SZ < 84349
        #define CACHEBLOCK_LENGTH 88
    #elif SMALLCACHE_SZ < 91761
        #define CACHEBLOCK_LENGTH 92
    #elif SMALLCACHE_SZ < 99485
        #define CACHEBLOCK_LENGTH 96
    #elif SMALLCACHE_SZ < 107521
        #define CACHEBLOCK_LENGTH 100
    #elif SMALLCACHE_SZ < 115869
        #define CACHEBLOCK_LENGTH 104
    #elif SMALLCACHE_SZ < 124529
        #define CACHEBLOCK_LENGTH 108
    #elif SMALLCACHE_SZ < 133501
        #define CACHEBLOCK_LENGTH 112
    #elif SMALLCACHE_SZ < 142785
        #define CACHEBLOCK_LENGTH 116
    #elif SMALLCACHE_SZ < 152381
        #define CACHEBLOCK_LENGTH 120
    #else
        #warning "Program isn't optimized for so large Cache"
        #define CACHEBLOCK_LENGTH 124
    #endif
#else
    #ifndef CACHEBLOCK_LENGTH
        #error "Unsupported SMALLCACHE_FRACTION"
    #endif
#endif

