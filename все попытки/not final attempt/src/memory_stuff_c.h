#include "memory_stuff.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32
    #include <malloc.h>
#endif

static inline aligned_realloc(aligned_double *ptr, size_t new_size, size_t old_size){
#ifdef _WIN32
    return _aligned_realloc(ptr, new_size, ALLIGNMENT);
#else
    aligned_double *possibly_new_ptr = realloc(ptr, new_size);
    assert(possibly_new_ptr);
    if ((uintptr_t)possibly_new_ptr % ALLIGNMENT){
        aligned_double *new_ptr;
        new_ptr = aligned_alloc(ALLIGNMENT, new_size);
        if (new_size > old_size){
            memcpy(new_ptr, possibly_new_ptr, old_size);
            memset(new_ptr+old_size, 0, new_size-old_size); // мб не нужно
        }
        else
             memcpy(new_ptr, possibly_new_ptr, new_size);
        
        free(possibly_new_ptr);
        possibly_new_ptr = new_ptr;
    }
    return possibly_new_ptr;
#endif
}

CF aligned_double* data_reserve(struct DataVector* const data, const size_t quantity){
    if (data->capacity < data->size + quantity){
        // Выделение памяти
        size_t new_capacity = (data->capacity*2 > data->size + quantity)?
                                data->capacity*2:
                                data->size+quantity;
        aligned_double* new_mem = aligned_realloc(
                                    data->doubles, 
                                    new_capacity * sizeof aligned_double,
                                    data->capacity * sizeof aligned_double
                                  );
        assert(new_mem);
        data->doubles = new_mem;
        data->capacity = new_capacity;
    } 
    aligned_double* ptr_to_elem_after_last = data->doubles + data->size;
    data->size += quantity;
    return ptr_to_elem_after_last;
}

CF void data_construct(struct DataVector* const data, const size_t capacity){
#ifdef _WIN32
    data->doubles = _aligned_malloc(ALLIGNMENT, capacity*sizeof aligned_double);
#else
    data->doubles = aligned_alloc(ALLIGNMENT, capacity*sizeof aligned_double);
#endif
    memset(data->doubles, 0, capacity*sizeof aligned_double);
    data->capacity = capacity;
    data->size = 0;
}

CF void data_destruct(struct DataVector* const data){
    free(data->doubles);
    data->capacity = 0;
    data->size = 0;
}

CF void data_shrink(struct DataVector* const data){
    data->doubles = aligned_realloc(
                                        data->doubles,
                                        data->size * sizeof aligned_double,
                                        data->capacity * sizeof aligned_double
                                   );
    assert(data->doubles);
    data->capacity = data->size;
}

//CF int calculate_ROWHEIGHT(const size_t nonzero, const size_t length, const size_t height){
//   
//}

